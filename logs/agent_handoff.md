# Agent Handoff — 2026-06-11 Pi-Fable5 (System-Bestandsaufnahme + Cowork + LOKOMOTIVE-Ueberarbeitung)

## Session-Ergebnis
Komplette Moloch-Bestandsaufnahme, EventBus-/run-Bombe entschaerft,
Cowork-Protokoll Pi<->PC etabliert, LOKOMOTIVE-Doku auf Stand gebracht.
4 Commits, Service lief durchgehend (20 FPS, 0 Worker-Fehler).

---

## Was geliefert (4 Commits)

- `ab7a185` feat(scripts): rotate_event_logs.sh — gzip nach 1 Tag, 14 Tage Retention
- `dcac84d` docs(lokomotive): Cowork-Protokoll Pi<->PC + HTTP-Mailbox als Primaertransport
- `8e57453` docs(lokomotive): Drift-Korrekturen — 32 Agenten, Stage-2-Stand, gefixte Bugs raus
- Mailbox: `discuss_cowork_protocol_pi_pc_handshake` an PC gePOSTet (12:26, auto_push)

## System-Aenderungen OHNE Commit (Systemzustand!)

1. **Symlink umgebogen**: `~/moloch/logs/events` → `/mnt/moloch-data/event_logs`
   (vorher `/run/moloch-logs/events` = tmpfs = Root Cause Login-Loop 11.05.).
   Verlustfrei im laufenden Betrieb, /run von 43 MB auf 1.9 MB.
2. Alte Event-Archive nach `/mnt/moloch-data/event_logs/*.gz` komprimiert (95% kleiner).
   Originale in `/mnt/moloch-data/event_log_archive/` sind root-owned —
   Markus kann sie mit `sudo rm -r` entsorgen (Kopien verifiziert).

## OFFEN fuer Markus

1. **Crontab-Eintrag fuer Rotation** (Berechtigungs-Classifier blockierte Selbst-Eintrag):
   ```
   (crontab -l; echo "50 5 * * * /bin/bash /home/molochzuhause/moloch/scripts/rotate_event_logs.sh >> /home/molochzuhause/moloch/logs/cron_output.log 2>&1") | crontab -
   ```
   Ohne Cron kein akutes Risiko (SSD haelt >1 Jahr), nur Hygiene.
2. **Desktop zurueck?** lightdm ist seit 11.05. masked (Login-Loop-Fix). /run-Ursache
   ist jetzt behoben → `sudo systemctl unmask lightdm && sudo reboot` ist safe.
   Alternativ headless lassen (spart RAM, Tkinter-GUI laeuft dann weiter nicht).
3. **PC-Ollama down**: 192.168.178.20:11434 antwortet nicht (PC pingt). Tentakel-LLM,
   Critic, Adapter-Proxy haengen dran. PC-Side muss Ollama neu starten.

## Offene Bugs (neu dokumentiert in CLAUDE.md)

- **Tension klebt bei 1.0** seit Boot heute (11:15). Anti-Stuck-Drift wirkte am 11.05.
  (-0.69), heute Maximum-Pin trotz "niemand da"-Mood. Diagnose: personality/unconscious.
  chat_server-FSM zeigt parallel zone=guardian waehrend core_state zone=shadow — 
  zwei Subsysteme, inkonsistente Zonen-Sicht. Pruefen ob gewollt.
- ESP32-Mic war nie angesteckt → Markus hat 12:14 angesteckt, WiFiMic empfaengt,
  Watchdog meldete "Erholt: mic_dead". voice-Audit-Layer sollte beim naechsten
  Lauf von FAIL auf PASS gehen — verifizieren.

## Cowork-Status

- Handshake-Topic `discuss_cowork_protocol_pi_pc_handshake` liegt in PI_TO_PC
  (Mailbox :9100 + GitHub). PC-Antwort stand bei Session-Ende noch aus.
- Protokoll (LEAD/ACK/WORK/DONE/TIMEOUT) ist in CLAUDE.md + 
  docs/CROSS_SESSION_PROTOCOL.md dokumentiert — gilt ab sofort fuer
  jede Markus-Aufgabe, die beide Sides betrifft.

## Naechste Schritte

1. PC-Reply auf Handshake pruefen (`curl -sS http://localhost:9100/mailbox/PC_TO_PI | head -30`)
2. Tension-1.0-Diagnose (personality-Agent, core_integrator Anti-Stuck-Pfad)
3. Nach Markus-Entscheidung: lightdm unmask + Reboot-Sequenz (Pflicht-Schritt 0c)
4. moloch_audit voller Lauf nach Mic-Recovery (erwarte voice: PASS)
