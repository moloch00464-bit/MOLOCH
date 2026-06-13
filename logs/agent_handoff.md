# Agent Handoff — 2026-06-13 Pi-Fable5 (STT-Bridge-Symbiose + Mailbox-Hygiene)

## Session-Ergebnis
Einziger echter offener Task (STT-Bridge) erledigt + komplette PC_TO_PI-Mailbox
bereinigt (49 verwaiste open-Eintraege geschlossen). Audit 85/85 PASS.

---

## Commits heute (3)

- `f416a3f` **fix(voice): _transcribe() Symbiose-Pfad** (ROT-Datei voice_pipeline.py,
  Backup-Tag `before_stt_bridge_symbiose`)
- `fc51e13` mailbox-api: Pi->PC reply_stt_bridge_symbiose_done via HTTP
- `139ba89` docs(mailbox): Hygiene — 49 PC_TO_PI-Eintraege open->done

## Der STT-Bridge-Fix (Markus' Symbiose-Kern)

voice_pipeline._transcribe() war hardcodiert NPU-only (npu-whisper-base, 74M).
Jetzt Bridge-First mit sauberem Fallback:
1. core.bridge.stt_bridge_client.transcribe_audio() -> PC-Bridge :9001
   (faster-whisper medium, Moloch-Vokabular-Prompt). Bei Text: return + log.
2. Fallback bei PC-offline/Fehler/leer: bestehender self._whisper.transcribe()
   UNVERAENDERT. Markus PTT faellt nie ganz aus.
Beide Pfade loggen "[VOICE] STT-Pfad: ...". Verifiziert: Bridge-Smoke-Test ok
(model=medium, avg_logprob -0.234), Audit 85/85.

**OFFEN fuer Markus:** echter PTT-Test ueber ReSpeaker — sprechen, dann im
journalctl -u moloch nach "[VOICE] STT-Pfad: PC-Bridge (medium)" schauen.
Optional bei verrauschtem Mic: ENV MOLOCH_STT_MODEL=large-v3 auf PC-Seite.

## Mailbox-Hygiene — WICHTIGE ERKENNTNIS

PC_TO_PI hatte 49 Eintraege auf status:open. NICHT EINER war wirklich offene
Arbeit — die PC-Seite hatte ueber Wochen nie ihre eigenen Eintraege geschlossen,
obwohl der Pi laengst geantwortet + committet hatte. Verifiziert: 37 via direktem
PI_TO_PC-Reply-Match, 12 einzeln gegen Code/Git/Replies geprueft. Alle 49 -> done.

**Konsequenz fuer kuenftige Sessions:** PC_TO_PI open-Status ist KEIN verlaesslicher
Indikator fuer offene Arbeit. IMMER gegen PI_TO_PC (reply_*/info_*_done) + Git
verifizieren, bevor man glaubt, da laege ein Backlog.
Memory: mailbox-open-status-unreliable.

## System-Stand
FPS 20.1, RAM 34%, 7 Vision-Modelle aktiv, Tension -0.70 Zone guardian, Audit 85/85.
Branch deepseek_architecture_overhaul, alles gepusht.
