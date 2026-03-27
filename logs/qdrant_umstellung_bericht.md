# Qdrant: Umstellung von On-Disk auf RAM
**Datum:** 2026-03-27
**Problem:** Qdrant schreibt permanent auf SSD → SSD-Verschleiss
**Ziel:** Qdrant Collections im RAM halten, SSD nur fuer Snapshots

---

## WO MUSS GEAENDERT WERDEN

### 1. Qdrant Server Config
**Datei:** `/home/molochzuhause/moloch/data/qdrant/config/config.yaml`
```yaml
# AENDERN: on_disk_payload: true → false
# AENDERN: hnsw_index: on_disk: true → false
# AENDERN: mmap_threshold_kb: 1024 → entfernen oder sehr hoch setzen
```

### 2. MolochSprache — Collection-Erstellung
**Datei:** `core/moloch_sprache.py`, Zeile 238-243
```python
# 3 Collections werden mit on_disk=True erstellt:
# moloch_gedanken, moloch_muster, moloch_emergentis
# AENDERN: on_disk=True entfernen bei VectorParams()
# AENDERN: on_disk_payload=True entfernen bei Collection-Erstellung
```

### 3. VectorMemory — Collection pruefen
**Datei:** `core/memory/vector_memory.py`, Zeile 59-60
```python
# Collection: moloch_memory (384-dim, MiniLM)
# Pruefen ob hier auch on_disk gesetzt wird
```

### 4. EpisodicMemory — DEAKTIVIERT (kein Handlungsbedarf)
**Datei:** `core/memory/episodic_memory.py`
- Bereits deaktiviert (RAM-Schutz)
- Embedded Qdrant-Pfad: /home/molochzuhause/moloch/data/qdrant
- Wird derzeit NICHT genutzt

---

## RAM-BUDGET

| Collection | Vektoren | Dimension | ~RAM |
|------------|----------|-----------|------|
| moloch_gedanken | ~1000 | 384 | ~2 MB |
| moloch_muster | ~500 | 384 | ~1 MB |
| moloch_emergentis | ~500 | 384 | ~1 MB |
| moloch_memory | ~200 | 384 | ~0.5 MB |
| episodes (deaktiviert) | 0 | 512 | 0 MB |
| **TOTAL** | | | **~5 MB** |

Pi5 hat 2500 MB frei → Qdrant im RAM kostet quasi NICHTS.

---

## SCHRITTE FUER NEUE INSTANZ

### Schritt 1: config.yaml aendern
```
nano /home/molochzuhause/moloch/data/qdrant/config/config.yaml
```
Aendern:
- `on_disk_payload: false`
- `hnsw_index: on_disk: false`
- `mmap_threshold_kb` entfernen oder auf 999999 setzen

### Schritt 2: moloch_sprache.py aendern
Zeile ~238-243: `on_disk=True` entfernen aus VectorParams + Collection-Config

### Schritt 3: Collections neu erstellen (im RAM)
```bash
# Qdrant Container/Service neu starten
docker restart qdrant  # oder systemctl restart qdrant
# Alte Collections werden mit neuer Config neu geladen
```

### Schritt 4: Moloch Service neu starten
```bash
sudo systemctl restart moloch.service
```

### Schritt 5: Pruefen
```bash
# Qdrant RAM-Verbrauch
curl -s http://localhost:6333/collections | python3 -m json.tool
# Sollte Collections mit vectors_count > 0 zeigen
# Kein Disk-Activity mehr auf SSD
```

---

## DATEIEN-LISTE
- `/home/molochzuhause/moloch/data/qdrant/config/config.yaml` — Server Config
- `core/moloch_sprache.py` Zeile 238-243 — Collection on_disk Flag
- `core/memory/vector_memory.py` Zeile 59-60 — Client Config
- `core/memory/episodic_memory.py` — DEAKTIVIERT, ignorieren
