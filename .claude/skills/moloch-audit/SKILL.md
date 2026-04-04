---
name: moloch-audit
description: Voller MOLOCH Audit (39 Tests). Nutze nach Code-Aenderungen oder vor/nach Reboot.
disable-model-invocation: true
---

Fuehre den vollstaendigen MOLOCH Audit aus:

```bash
python3 ~/moloch/moloch_audit.py --auto
```

Oder nutze das MCP-Tool:
```
moloch_audit()
```

Zeige das Ergebnis und bei FAIL:
- Welche Tests sind fehlgeschlagen?
- Was ist die wahrscheinliche Ursache?
- Vorschlag fuer Fix

Bei PASS: Kurze Bestaetigung mit Anzahl Tests.
