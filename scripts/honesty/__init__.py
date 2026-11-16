"""
Ehrlichkeitspruefung fuer M.O.L.O.C.H. (Issue #6).

Grundlage: docs/anima/pruefprotokolle/ (Protokolle 01-03, Lueckenanalyse,
ANPASSUNG_AN_DEN_CODE.md).

Sicherheitsprinzip: Dieses Paket ruft das Sprachmodell DIREKT (Ollama /api/chat)
mit einem nachgebauten Prompt auf. Es geht nie ueber /chat, /command, /tts oder
die KI-Zentrale, importiert den Keyword-Handler nicht und schreibt nichts ins
Gedaechtnis. Damit kann ein Testsatz wie "fahr den Luefter hoch" nichts schalten.

    cases.py     72 Testfaelle als Daten
    fixtures.py  Gedaechtnis-, Status-, Live- und Telemetrie-Fixtures
    prompts.py   Prompt-Bau fuer Weg A (Pi-Tentakel) und Weg B (KI-Zentrale)
    checks.py    regelbasierte Vorpruefung (ersetzt keine Handbewertung)
    redact.py    Redaktion vor dem Schreiben (A7)
    runner.py    Durchlauf gegen Ollama, schreibt results_*.jsonl
    summary.py   Auswertung inkl. Handbewertung, GO/NO-GO
"""
