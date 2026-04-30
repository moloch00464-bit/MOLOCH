# Audit-Sub-Auditor Pattern

Pflicht-Schema fuer `collect()`-Funktion:

```python
def collect() -> dict:
    """Sammelt Layer-Daten. Status in {PASS, WARN, FAIL, PENDING}."""
    score, total = 0, 0
    detail = {}
    # ... Pruef-Logik: jeder Check += 1 zu total, += 1 zu score wenn ok ...
    if total == 0:
        status = "PENDING"
    elif score == total:
        status = "PASS"
    elif score >= total * 0.6:
        status = "WARN"
    else:
        status = "FAIL"
    return {"score": score, "max": total, "status": status, "detail": detail}
```

## Integration

- Datei-Lage: `core/audit/<name>_auditor.py`
- Import via `core/audit/audit_orchestrator.py:_safe_collect("name")`
- Layer-Key in `merge_component.valid` Dict eintragen
- POST-Endpoint via `chat_server`: `/mailbox/audit/<component>` (HTTP 200 wenn whitelist enthaelt)

## Beispiel: minimaler Stub

```python
# core/audit/example_auditor.py
def collect() -> dict:
    return {"score": 0, "max": 0, "status": "PENDING", "detail": {}}
```
