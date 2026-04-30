# Atomic JSON-Write (NEVER 6)

`open(path, "w")` direkt = Race-Condition: Reader sieht halbe Datei.

## Pflicht-Snippet

```python
import json, os, tempfile
from pathlib import Path

def atomic_write_json(path: Path, data: dict) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, str(path))
        return True
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False
```

## Pflicht fuer

- `/dev/shm/audit_state.json`
- `logs/*.json`
- `character_journal` events
- Jeden persistenten State der parallel gelesen wird

## Verifikation

`os.replace(tmp, dst)` ist atomar auf POSIX (rename(2)). Reader sehen
ENTWEDER alte ODER neue Version, nie eine halbe.

Auf Windows: gleiches Verhalten via `MoveFileEx(MOVEFILE_REPLACE_EXISTING)`.
