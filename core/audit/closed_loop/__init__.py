"""Closed-Loop-Verifier (Welle 15).

Maturitaetsstufe L3: Befehl -> Sensor -> Effekt verifiziert.

Pro Verifier:
  verify(timeout_s) -> {score, max, status, command_sent, baseline, after, delta,
                        duration_s, detail}

Status: PASS | WARN | FAIL | SKIP

Best-effort imports — Verifier crasht NIE, returnt SKIP wenn Aktor nicht verfuegbar.
Cleanup nach jedem Test (Aktor auf vorigen Zustand zuruecksetzen).
"""
