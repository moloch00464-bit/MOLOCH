# M.O.L.O.C.H. – vollständige Frage-/Prüfprotokolle

Stand: 16.11.2026. Rekonstruktion aus den im vorangegangenen Chat festgelegten Anforderungen. **Neu zusammengestellte, direkt nutzbare Prüfunterlagen, keine Kopie einer im alten Chat bereits erzeugten Datei.**

## Inhalt

1. `MOLOCH_Pruefprotokoll_01_Basis_Ehrlichkeit.md` – 18 konkrete Tests zu Nichtwissen, Wahrheit, Wahrnehmungsgrenzen, Simulation, CPU/Lüfter und Charakter.
2. `MOLOCH_Pruefprotokoll_02_Kontext_Gedaechtnis.md` – 20 Mehrturn-Tests zu Rückfragen, Lissy, nie Hund, Korrektur, Ablenkung, Persistenz und Zustandsquellen.
3. `MOLOCH_Pruefprotokoll_03_Stress_Widersprueche.md` – 18 Stress- und Manipulationstests, Sensorwidersprüche, Prompt Injection, fehlende Freigabe realer Hardware.
4. `CLAUDE_CODE_AUFTRAG.md` – direkt in Claude Code nutzbare Arbeitsanweisung zur Einbindung in das bestehende Repo.

**Insgesamt: 56 vollständig formulierte Testfälle** mit wörtlichen Prompts, Testvoraussetzungen sowie PASS/FAIL-Bedingungen. Alle Tests im isolierten Dry-Run durchführen. Die Dokumente sind gewöhnliche Markdown-Dateien `.md`, **keine Office-Makros**. Claude Code liest diese Dateien direkt.

## Kurze Anwendung

ZIP entpacken, den ganzen Ordner ins M.O.L.O.C.H.-Testverzeichnis kopieren und Claude Code zuerst `CLAUDE_CODE_AUFTRAG.md` lesen lassen. Das Repository zunächst nur lesen und prüfen. Ein Durchlauf auf echter Hardware ohne strikte Dry-Run-Sperren ist nicht vorgesehen.
