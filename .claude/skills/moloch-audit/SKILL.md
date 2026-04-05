---
name: moloch-audit
description: Voller MOLOCH Audit (54 Tests). Nutze nach Code-Aenderungen oder vor/nach Reboot. FAIL = sofort stoppen.
disable-model-invocation: true
---

# MOLOCH Audit — 54 Tests

Nutze das MCP-Tool:
```
moloch_audit()
```

**Erwartetes Ergebnis: 54/54 PASS**

Bei FAIL:
- Welche Tests fehlgeschlagen?
- Wahrscheinliche Ursache (letzte Code-Aenderung?)
- Fix vorschlagen
- `git checkout -- [datei]` wenn noetig

Bei PASS:
- Kurze Bestaetigung: "Audit 54/54 PASS"
- Weiter mit naechstem Schritt

**Regel: Bei FAIL nicht weitermachen — erst fixen!**
