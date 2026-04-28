PHASE 0a — DIAGNOSE-BERICHT (2026-04-28 07:50)
================================================

BUG 1 (Landmarks weg): PHANTOM — kein Fix nötig
-------------------------------------------------
- NPU Worker: alle running mit Inferenzen
  Face: 2691 inf, 0 errors | Pose: 1808 inf | ReID: 1085 inf | Depth: 542 inf
- panel_detections (3 Einträge): Daten komplett vorhanden
  [0] class=face: landmarks SET (5 Punkte), face_id=Markus, sim=0.56
  [1] class=pose: keypoints SET (17 Punkte)
  [2] class=pose: keypoints SET
- Erstes Sample zeigte nur class="person" (per Design ohne Landmarks)
  → Markus hat vermutlich direkt nach Restart oder Panel-Cache gesehen

BUG 2 (Tension stuck Guardian): REAL, aber komplexer
-----------------------------------------------------
- Aktuell: tension=-1.0 (saturated low, Range [-1, 1]), zone=guardian
- Markus erkannt: face_id=markus, sim=0.56 → owner_confirmed=true → konsistent
- Bug-Pfad: unknown_person-Awareness-Input wird zwar gesetzt (moloch_service.py:673
  via feed_event), aber im tension_integrator existiert KEIN direkter
  "face_id=None → tension++" Push.
- on_activity_changed nutzt unknown_person nur bei dominance_delta<0
- on_context_update mappt nur conflict_input / respect_score
- → asymmetrisch: Markus → dominance↑ klar definiert; Unknown → kein direkter
  Tension-Schub, nur indirekt via Activity

EMPFEHLUNG:
- 0b: SKIP (kein Bug)
- 0c: in tension_integrator.py NEUE Methode on_face_recognized() oder Erweiterung
  von on_context_update, die bei face_id ∈ {None, "unknown", "Unbekannt"} und
  person_detected=True einen direkten tension_impulse +0.15 sendet.
  Subscriber-Pfad: face_recognized-Event aus moloch_service.py / face_pipeline.py.
