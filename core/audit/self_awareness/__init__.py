"""Self-Awareness Subpackage — Welle 17 (L5).

Liefert Moloch ein Bewusstsein darueber, was er gerade kann und was nicht.

Quelle: `/dev/shm/audit_state.json` (Audit-Orchestrator-Aggregat) +
`drift_events` der letzten 24h.

Zwei Module:
- `capability_inventory` — was Moloch JETZT kann ("ich kann sehen") /
  nicht kann ("PTZ tot") / nur eingeschraenkt ("Spotify-Token alt").
- `failure_reflection` — Reflexion ueber wiederkehrende Ausfaelle, in
  Moloch's Stimme (kurze deutsche Saetze fuer Chat/LLM-System-Prompt).

Beide Module sind best-effort: sie crashen nie, sondern liefern bei
fehlenden Daten leere/PENDING-Antworten zurueck.

Kein Edit am Audit-Orchestrator. Keine externen Schreibwege. Lesbar.
"""

from __future__ import annotations

from .capability_inventory import collect_capabilities
from .failure_reflection import reflect_on_failures

__all__ = ["collect_capabilities", "reflect_on_failures"]
