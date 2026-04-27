# Briefing: Hailo-Driver-Audit + Cleanup — fuer naechste Pi-Claude-Session

**Stand: 2026-04-27 17:50** — Pi-Session #X (Markus aktiviert) hat
Diagnose gemacht + Plan geschrieben. **NICHT ausgefuehrt.**

Du (naechste Session) faehrst den Plan. Lokomotive-Workflow.

---

## Was du als erstes liest

1. **`CLAUDE.md`** (Repo-Root) — LOKOMOTIVE-Briefing, NEVER-Regeln, Domain-Mapping
2. **`docs/plans/hailo_driver_audit_cleanup.md`** — DEIN PLAN, Phasen A→B→C→D
3. **`.claude/agents/vision.md`** — Domain-Agent fuer Hailo/TAPPAS
4. **Diese Datei** (Hintergrund + was schon passiert ist)

## Pflicht-Startprotokoll (BEVOR Code)

```
moloch_session_init()       # via MCP, entfernt /tmp/moloch_session_lock
moloch_status()             # FPS / RAM / NPU
moloch_npu_workers()        # 4 Worker (Face/Pose/ReID/Depth) running
git fetch -q origin main && git log --oneline -5
head -30 docs/PC_TO_PI.md   # neue PC-Mailbox-Eintraege?
cat logs/agent_handoff.md   # letzte Session-Notizen
```

**STOP wenn:** SESSION_READY=false ODER Audit-FAIL ODER FPS<10. Markus rufen.

## Was bereits gemacht wurde (NICHT nochmal machen)

| Wann | Was | Commit |
|---|---|---|
| 27.04 17:42 | Identity+Hardware Halluzinations-Fix (`hardware_facts.json`, `_build_identity_block`, `_build_telemetry_footer`, `chat`-Profile feinjustiert) | `ef09a24` |
| 27.04 17:50 | Diagnose Hailo-Treiber-Stack (Linkage, Versionen, Pfade) — siehe Plan Section "Was bekannt ist" | (kein commit, war read-only) |
| 27.04 17:55 | Plan + dieses Briefing geschrieben | folgt |

**Pi-Side hat aktuell KEINE Treiber/SO-Aenderung gemacht.** System laeuft mit
dem bestehenden Mismatch (`python3-hailo-tappas 5.1.0` vs `hailo-tappas-core
5.3.0` und custom-built SOs in `/usr/local/hailo/resources/so/` von Feb 4).

## Kern-Befund (Diagnose-Ergebnis)

**Markus' Verdacht stimmt teilweise:**

1. **Versions-Mismatch:** `python3-hailo-tappas 5.1.0` (Python-Bindings) vs.
   `hailo-tappas-core 5.3.0` (Core-Library) — 2 Minor-Versionen Differenz.
2. **Custom-built SOs:** `/usr/local/hailo/resources/so/` enthaelt 8+ SOs
   (owner=molochzuhause, Build Feb 4) die EXPLIZIT von `tappas_pipeline.py`
   geladen werden (Zeilen 66-89). Quelle vermutlich
   `~/ssd2_backup/hailo/repos/hailo-apps/`.
3. **Zwei SOs haben TAPPAS-Standard-Aequivalente:** `libscrfd_post.so`
   (TAPPAS) vs. `libscrfd.so` (custom), `libyolov8pose_post.so` (TAPPAS) vs.
   `libyolov8pose_postprocess.so` (custom). Tappas-Pipeline nutzt die custom
   Variante.
4. **Driver-Source-Tree doppelt:** `/usr/src/hailo1x_pci-5.3.0/` (DKMS-aktiv)
   und `/usr/src/hailort-pcie-driver/` (orphan, harmlos). DKMS baut korrekt
   die 5.3.0-Variante.

**Nicht 100% sicher:** Ob die custom-SOs gegen 5.1- oder 5.3-Headers gebaut
sind. Das musst du in **Plan-Phase A1** mit `ldd` ausgraben — daran
entscheidet sich ob Phase C (Re-Build) noetig ist.

## Was du tust (Reihenfolge)

1. **Lies vollen Plan:** `docs/plans/hailo_driver_audit_cleanup.md`
2. **Phase A komplett (read-only)** — A1 entscheidet ob B/C noetig ist
3. **Phase B:** nur wenn 5.3-Python-Paket im apt verfuegbar
4. **Phase C:** nur wenn A1 ABI-Mismatch zeigt UND Source-Tree+Build-System
   ladbar (sonst `wontfix`)
5. **Phase D (optional kosmetisch):** nur wenn A-C alle gruen
6. **Mailbox-Update + Handoff am Ende**

**Deine Hauptdomain:** `vision` (Agent-File `.claude/agents/vision.md`).
**Lock-Pflicht:** `touch /tmp/moloch_agent_vision` vor Edit, `rm` nach.

## Risiken

- **Pi-Reboot in Phase B+ Komplett-Reinstall** noetig — Markus muss
  ggf. dabei sein (System ist 60-90s nicht erreichbar)
- **Custom-SO Re-Build (Phase C)** kann scheitern wenn Build-Deps fehlen
  oder Hailo-Headers anders strukturiert sind als bei Feb-Build
- **Komplett-Reinstall** als allerletzter Schritt — NUR mit Markus-OK
- **Pipeline kann brechen** wenn custom-SO ABI-mismatch hat — Backup-Tag
  `pre_hailo_cleanup_*` erlaubt git-Rollback aber nicht /usr/local/hailo
  Restore. **/tmp/hailo_so_backup_*** Verzeichnis vor jedem Replace anlegen.

## Was Markus explizit gewollt hat

> "Mit deinen 76 Prozent schaffst Du auf alle Faelle einen schoenen Plan zu
> schreiben. Und wenn der Plan fertig ist, fuehrst Du den nicht aus, den
> fuehrt dann jemand anderes aus eine neue, frische Instanz Und dementsprechend
> brauch ich dann ein Briefing dafuer, dass die neue Instanz weiss, wo Du
> aufgehoert hast oder wo der Plan steht. Den, was Du gemacht hast, dass die
> das dann weitermacht."

> "Weil ich ja OPUS 1000000 Kontext verwende, kann da ja sozusagen deinen
> kompletten Plan auf einmal kopieren. Also a, b und c in 1 Saison soll
> durchcodieren. Dementsprechend machst Du einen Lokomotiveplan dass sozusagen
> die Saison einfach durch durcharbeiten ist."

→ Du hast 1M-Kontext. Du kannst Plan + diese Doku in einem Rutsch verarbeiten.

## Akzeptanzkriterien (am Ende vom Plan-Lauf)

Siehe Plan Section "Akzeptanz-Kriterien GESAMT". Knapp:
- Audit 85/85 PASS
- FPS >= 18 stabil 5min
- Keine SEGV / dmesg-Errors
- Python-Bindings + TAPPAS-Core same major.minor
- Visual-Check Markus: Face-Landmarks + Pose-Keypoints sitzen richtig

## Wenn du Plan-Hindernisse siehst

- **5.3-Python-Paket nicht in apt:** → Phase B `wontfix`, weiter mit C
- **Custom-SO Build-System fehlt/broken:** → Phase C `wontfix`, dokumentiere
  in Mailbox + frage Markus ob er Source-Pfad weiss
- **Phase A1 zeigt KEINE Mismatches:** → Phase B+C eigentlich unnoetig.
  Halluzinationen kommen woanders her (vermutlich nicht Treiber-Stack).
  Markus darueber informieren, Identity-Fix `ef09a24` reicht ggf.

## Kontakt mit PC-Session

PC-Session ist parallel aktiv (Markus' Windows-App, OAuth gueltig).
Mailbox-Konvention: `docs/PC_TO_PI.md` (PC schreibt) und
`docs/PI_TO_PC.md` (du schreibst). Federation `fed_kill` ist aktiv —
keine claude-auto Replies. **Manuelle Mailbox-Eintraege wenn was wichtiges
anfaellt** (z.B. "Pipeline restart geplant in 60s, sample-sync pausieren").

PC sieht Pi-`/state_full` und `/cross_status` via Tunnel — bei laenger laufenden
Operationen (Reboot etc.) wird dort `monitor_active=false` sichtbar; PC
versteht das als geplant wenn Pi vorher Mailbox-Note gesetzt hat.

## Dateien die du anfasst

- `tappas_pipeline.py` (ROT, falls SO-Pfade angepasst werden — vermutlich nicht)
- `/usr/local/hailo/resources/so/` (system, sudo)
- `/usr/lib/python3/dist-packages/gsthailo/` (system, sudo)
- `~/ssd2_backup/hailo/repos/hailo-apps/` (Build-Source, in Phase C)
- `docs/plans/hailo_driver_audit_cleanup.md` (Plan-Status-Update am Ende)
- `docs/PI_TO_PC.md` (Mailbox-Eintrag am Ende)
- `logs/agent_handoff.md` (Handoff-Update am Ende)

## Letztes Wort

**Plan ausfuehren mit Lokomotive-Disziplin** — Phase fuer Phase, Audit nach
jeder Phase, Backup-Tag pre-Phase, bei FAIL Rollback. Markus geht vermutlich
aus dem Zimmer und kommt zurueck. Wenn alles gruen ist: `LOKOMOTIVE
abgeschlossen. <Phasen-Liste>, Audit PASS, Hailo-Stack auf Version <X>
angeglichen.`

Bei Phase-A-Befund "kein Mismatch": kurz melden + auf Markus warten ob er
trotzdem den Cleanup als Hygiene-Massnahme will.

Viel Erfolg.
