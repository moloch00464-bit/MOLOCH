# Agent Handoff — 2026-06-11 Pi-Fable5 (Nacht-Session: Desktop + Input-TTL-Fix)

## Session-Ergebnis
Desktop wiederhergestellt (ohne Reboot), Dominance-Pin-Root-Cause gefunden
und gefixt (Input-TTL in core_integrator.py), Audit 85/85 PASS.
Tension reagiert wieder dynamisch: 0.64 -> -0.69 binnen 2.5 min, Zone guardian.

---

## Commits heute (gesamt 6)

- `ab7a185` feat(scripts): rotate_event_logs.sh (gzip 1 Tag, Retention 14 Tage)
- `dcac84d` docs(lokomotive): Cowork-Protokoll Pi<->PC + HTTP-Mailbox primaer
- `8e57453` docs(lokomotive): Drift-Korrekturen 32 Agenten, Stage-2, Bugs
- `14507c3` docs(handoff): Mittags-Stand
- `c0e4f8b` **fix(core): Input-TTL 120s — Dominance-Pin geloest** (ROT-Datei,
  Rollback-Tag: `before_input_ttl_fix`)

## Der Input-TTL-Fix (WICHTIG fuer naechste Sessions)

Root Cause: `update_input()` speicherte Inputs ohne Verfall. Unconscious-
Mood-Impulse (conflict_input/unknown_person/markus_recognized) klebten ewig
-> Dauer-Impuls schlug DOMINANCE_DRIFT_RATE um Faktor 6-12 -> Dominance
pinnte bei -0.995 (mittags) bzw. +1.000 (abends), Phantom-unknown_person.
Fix: Timestamp pro (source,key), Verfall nach 120s ohne Refresh im Tick.
`alarm_active` exempt. Logging: "[CORE] Input verfallen: source.key".

## System-Aenderungen ohne Commit

1. **lightdm unmasked + enabled + gestartet** — Desktop laeuft seit 23:19
   OHNE Reboot (der alte verklemmten-Greeter-Grund war durch den Boot
   von heute Mittag obsolet). Markus muss am HDMI-Monitor verifizieren.
2. Symlink `logs/events` -> `/mnt/moloch-data/event_logs` (von Mittags-Session).

## OFFEN fuer Markus

1. **Crontab-Eintrag** (Classifier blockiert Selbst-Eintrag, 2x versucht):
   ```
   (crontab -l; echo "50 5 * * * /bin/bash /home/molochzuhause/moloch/scripts/rotate_event_logs.sh >> /home/molochzuhause/moloch/logs/cron_output.log 2>&1") | crontab -
   ```
2. Alte root-owned Archive entsorgen: `sudo rm -r /mnt/moloch-data/event_log_archive`
   (gzip-Kopien liegen verifiziert in /mnt/moloch-data/event_logs/)

## Offene Punkte (naechste Session)

1. **Zonen-Triple-Split**: DREI Zustandsmaschinen fuehren je eigene zone:
   core_integrator (Source of Truth), personality/state_engine (eigene FSM —
   speist /api/state/current im chat_server!), ArbitrationEngine (Override-Layer).
   Cockpit zeigte "guardian" waehrend Core "shadow" sagte. Fix: state_engine
   soll Zone vom CoreIntegrator uebernehmen. Territorium: personality-Agent.
   Fundstellen: chat_server.py:2926 (get_state_engine().tick), core_integrator
   get_personality_zone().
2. **owner_confirmed-Timer nicht persistiert** (core_integrator start/persist):
   nach Boot ist owner_confirmed weg obwohl dominance persistiert wurde.
   Mit Input-TTL weniger kritisch — beobachten, ob noch noetig.
3. **PC-Session-Handshake unbeantwortet**: discuss_cowork_protocol_pi_pc_handshake
   (12:26) liegt in PI_TO_PC. PC-Ollama wurde drueben neu gestartet (Tentakel
   wieder ok), aber Mailbox ungelesen. Markus: PC-Session auf Mailbox stossen.
4. voice-Audit-Layer Recovery nach Mic-Anstecken verifizieren (Watchdog
   meldete "Erholt: mic_dead" um 12:17 — grosser Audit lief 23:25 mit PASS).

## Service-Status bei Uebergabe
moloch.service aktiv, 20 FPS, Audit 85/85 PASS, Tension -0.69 (guardian),
lightdm aktiv, PC-Tentakel erreichbar, ESP32-Mic streamt.
