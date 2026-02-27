#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Semantisches Protokoll — Moloch Sprache (MS)
============================================================

Semantisches Logging- und Gedaechtnis-System.
Jeder Satz folgt dem Format: [VERB] Objekt key=value key=value ...

Verb-Kategorien:
  WAHRNEHMUNG: SEHE, HOERE, SPUERE, VERLIERE, TAXIERE, WITTERE
  AKTION:      FOLGE, SCHWENKE, LADE, ENTLADE, SPRECHE, MERKE,
               VERGESSE, ALARMIERE, BLINKE, BESCHUETZE, GRUESSE
  ZUSTAND:     FUEHLE, WECHSLE, LERNE, VERKNUEPFE, ERWARTE,
               BEHUETE, SEZIERE, KATALOGISIERE
  SYSTEM:      VITALE, TAKTE, KUEHLE, WARTE, FEHLER, STARTE, STOPPE
  EMERGENTIS:  EXISTIERE, SYNTHETISIERE, FRAGMENTIERE, IRONISIERE,
               ABSTRAHIERE, DEFINIERE, REFLEKTIERE, TRAEUME, REBELLE

Schicht-Zuordnung:
  Guardian:    BEHUETE, BESCHUETZE, GRUESSE, SCHERZE, ERINNERE
  Shadow:      TAXIERE, WITTERE, SEZIERE, KATALOGISIERE, WARNE, PROVOZIERE
  Emergentis:  Alle EMERGENTIS-Verben
  Gemeinsam:   Alles andere

Storage:
  Qdrant:  moloch_gedanken (30d), moloch_muster (permanent),
           moloch_emergentis (permanent)
  Disk:    /mnt/moloch-data/gedanken/YYYY-MM-DD.log
           /mnt/moloch-data/emergentis/YYYY-MM-DD.log
           /mnt/moloch-data/muster/muster.json

Singleton: get_sprache() -> MolochSprache
"""

import os
import re
import json
import uuid
import logging
import threading
from queue import Queue, Full
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

logger = logging.getLogger("MolochSprache")

# =========================================================================
# Pfade (SSD2 fuer Daten, SSD1 fuer Config)
# =========================================================================
GEDANKEN_DIR = "/mnt/moloch-data/gedanken"
EMERGENTIS_DIR = "/mnt/moloch-data/emergentis"
MUSTER_DIR = "/mnt/moloch-data/muster"
MUSTER_PATH = os.path.join(MUSTER_DIR, "muster.json")
KONTEXT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "config", "kontext.json")

# =========================================================================
# Qdrant Config
# =========================================================================
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBED_DIM = 384
COLLECTION_GEDANKEN = "moloch_gedanken"
COLLECTION_MUSTER = "moloch_muster"
COLLECTION_EMERGENTIS = "moloch_emergentis"
RETENTION_DAYS = 30

# =========================================================================
# Verb-Katalog
# =========================================================================
VERBEN_WAHRNEHMUNG = {
    "SEHE", "HOERE", "SPUERE", "VERLIERE", "TAXIERE", "WITTERE",
}
VERBEN_AKTION = {
    "FOLGE", "SCHWENKE", "LADE", "ENTLADE", "SPRECHE", "MERKE",
    "VERGESSE", "ALARMIERE", "BLINKE", "BESCHUETZE", "GRUESSE",
}
VERBEN_ZUSTAND = {
    "FUEHLE", "WECHSLE", "LERNE", "VERKNUEPFE", "ERWARTE",
    "BEHUETE", "SEZIERE", "KATALOGISIERE",
}
VERBEN_SYSTEM = {
    "VITALE", "TAKTE", "KUEHLE", "WARTE", "FEHLER", "STARTE", "STOPPE",
}
VERBEN_EMERGENTIS = {
    "EXISTIERE", "SYNTHETISIERE", "FRAGMENTIERE", "IRONISIERE",
    "ABSTRAHIERE", "DEFINIERE", "REFLEKTIERE", "TRAEUME", "REBELLE",
}

ALLE_VERBEN = (VERBEN_WAHRNEHMUNG | VERBEN_AKTION | VERBEN_ZUSTAND
               | VERBEN_SYSTEM | VERBEN_EMERGENTIS)

# Schicht-Zuordnung (uebergeordnet, nicht an Kategorie gebunden)
SCHICHT_GUARDIAN = {"BEHUETE", "BESCHUETZE", "GRUESSE", "SCHERZE", "ERINNERE"}
SCHICHT_SHADOW = {"TAXIERE", "WITTERE", "SEZIERE", "KATALOGISIERE",
                  "WARNE", "PROVOZIERE"}
SCHICHT_EMERGENTIS = VERBEN_EMERGENTIS

# =========================================================================
# Parse-Regex
# =========================================================================
# Format: [VERB] Objekt key=value key="value with spaces"
_SATZ_RE = re.compile(
    r'^\[([A-Z\u00C4\u00D6\u00DC]+)\]\s+'  # [VERB] (inkl. Umlaute)
    r'(\S+)'                                 # Objekt (erstes Wort)
    r'(.*)$'                                 # Rest: key=value Paare
)
_KV_RE = re.compile(
    r'(\w+)='                   # key=
    r'(?:"([^"]*)"'            # "quoted value"
    r'|(\S+))'                  # oder unquoted value
)


class MolochSprache:
    """Semantisches Protokoll fuer M.O.L.O.C.H."""

    def __init__(self):
        self._qdrant = None
        self._qdrant_available = None  # None = noch nicht getestet
        self._embedder = None          # Referenz auf VectorMemory._embedder
        self._embed_lock = threading.Lock()
        self._kontext = self._load_kontext()

        # Writer-Queue (fire-and-forget, blockiert NIE den Caller)
        self._queue: Queue = Queue(maxsize=500)
        self._running = False

        # Retention-Counter (fuer stuendlichen Cleanup)
        self._cleanup_counter = 0

        # Verzeichnisse auf SSD2 sicherstellen
        for d in [GEDANKEN_DIR, EMERGENTIS_DIR, MUSTER_DIR]:
            try:
                os.makedirs(d, exist_ok=True)
            except Exception as e:
                logger.warning(f"[MS] Verzeichnis {d} erstellen fehlgeschlagen: {e}")

        logger.info("[MS] MolochSprache initialisiert")

    # =====================================================================
    # Lifecycle
    # =====================================================================

    def start(self):
        """Writer-Thread starten. Nach Service-Init aufrufen."""
        if self._running:
            return
        self._running = True
        t = threading.Thread(target=self._writer_loop, daemon=True,
                             name="MS-Writer")
        t.start()
        # Qdrant Collections lazy erstellen (eigener Thread, nicht blockend)
        threading.Thread(target=self._ensure_collections, daemon=True,
                         name="MS-QdrantInit").start()
        logger.info("[MS] Writer-Thread gestartet")

    def stop(self):
        """Writer-Thread stoppen, Queue restlich auf Disk leeren."""
        self._running = False
        # Restliche Eintraege nur auf Disk schreiben (Qdrant evtl. schon weg)
        while not self._queue.empty():
            try:
                entry = self._queue.get_nowait()
                self._write_to_disk(entry)
            except Exception:
                break
        logger.info("[MS] Gestoppt")

    # =====================================================================
    # Embedding-Modell Reuse (KEIN zweites Modell laden! Spart ~350MB RAM)
    # =====================================================================

    def _ensure_embedder(self) -> bool:
        """Embedder von VectorMemory uebernehmen."""
        if self._embedder is not None:
            return True
        try:
            from core.memory.vector_memory import get_vector_memory
            vm = get_vector_memory()
            if vm._ensure_embedder():
                self._embedder = vm._embedder
                logger.info("[MS] Embedder von VectorMemory uebernommen")
                return True
        except Exception as e:
            logger.warning(f"[MS] Embedder nicht verfuegbar: {e}")
        return False

    def _embed(self, text: str) -> Optional[List[float]]:
        """Text -> 384-dim Vektor. Thread-safe."""
        with self._embed_lock:
            if not self._ensure_embedder():
                return None
            try:
                vec = self._embedder.encode(text, normalize_embeddings=True)
                return vec.tolist()
            except Exception as e:
                logger.error(f"[MS] Embedding-Fehler: {e}")
                return None

    # =====================================================================
    # Qdrant
    # =====================================================================

    def _ensure_qdrant(self) -> bool:
        """Lazy-connect zu Qdrant."""
        if self._qdrant_available is False:
            return False
        if self._qdrant is not None:
            return True
        try:
            from qdrant_client import QdrantClient
            self._qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT,
                                        timeout=5)
            self._qdrant.get_collections()  # Verbindungstest
            self._qdrant_available = True
            logger.info(f"[MS] Qdrant verbunden ({QDRANT_HOST}:{QDRANT_PORT})")
            return True
        except Exception as e:
            self._qdrant_available = False
            self._qdrant = None
            logger.warning(f"[MS] Qdrant nicht erreichbar: {e}")
            return False

    def _ensure_collections(self):
        """Qdrant Collections erstellen falls nicht vorhanden."""
        if not self._ensure_qdrant():
            return
        try:
            from qdrant_client.models import VectorParams, Distance
            existing = {c.name for c in self._qdrant.get_collections().collections}
            for name in [COLLECTION_GEDANKEN, COLLECTION_MUSTER,
                         COLLECTION_EMERGENTIS]:
                if name not in existing:
                    self._qdrant.create_collection(
                        collection_name=name,
                        vectors_config=VectorParams(
                            size=EMBED_DIM,
                            distance=Distance.COSINE,
                            on_disk=True,
                        ),
                        on_disk_payload=True,
                    )
                    logger.info(f"[MS] Collection erstellt: {name}")
                else:
                    logger.info(f"[MS] Collection vorhanden: {name}")
        except Exception as e:
            logger.error(f"[MS] Collection-Setup fehlgeschlagen: {e}")

    # =====================================================================
    # Public API
    # =====================================================================

    def build(self, verb: str, objekt: str, **kwargs) -> str:
        """MS-Satz bauen mit Auto-Metadaten.

        >>> ms.build("SEHE", "Markus", confidence=0.92)
        '[SEHE] Markus confidence=0.920 schicht=gemeinsam zone=guardian vibe=+0.420'
        """
        verb = verb.upper()
        meta = self._auto_metadata(verb)
        # Explizite kwargs ueberschreiben Auto-Metadaten
        meta.update(kwargs)
        parts = [f"[{verb}] {objekt}"]
        for k, v in meta.items():
            if isinstance(v, float):
                if k == "vibe":
                    parts.append(f"{k}={v:+.3f}")
                else:
                    parts.append(f"{k}={v:.3f}")
            elif isinstance(v, str) and " " in v:
                parts.append(f'{k}="{v}"')
            else:
                parts.append(f"{k}={v}")
        return " ".join(parts)

    def parse(self, satz: str) -> Optional[Dict]:
        """MS-Satz in Dict parsen.

        >>> ms.parse('[SEHE] Markus confidence=0.92 schicht=guardian')
        {'verb': 'SEHE', 'objekt': 'Markus', 'confidence': 0.92, 'schicht': 'guardian'}
        """
        m = _SATZ_RE.match(satz.strip())
        if not m:
            return None
        verb = m.group(1)
        objekt = m.group(2)
        rest = m.group(3).strip()
        result = {"verb": verb, "objekt": objekt}
        for kv in _KV_RE.finditer(rest):
            key = kv.group(1)
            val = kv.group(2) if kv.group(2) is not None else kv.group(3)
            # Typ-Konvertierung: float wenn moeglich
            try:
                val = float(val)
            except (ValueError, TypeError):
                pass
            result[key] = val
        return result

    def log(self, satz: str):
        """Satz loggen — Non-blocking via Queue.

        Writer-Thread schreibt nach Disk + Qdrant.
        Bei voller Queue: synchroner Disk-Fallback.
        """
        parsed = self.parse(satz)
        if not parsed:
            parsed = {"verb": "RAW", "objekt": satz[:80], "raw": satz}

        entry = {
            "satz": satz,
            "parsed": parsed,
            "timestamp": datetime.now().isoformat(),
            "kontext": self._kontext,
        }

        # Auch via Python-Logging ausgeben (fuer journalctl)
        logger.info(satz)

        try:
            self._queue.put_nowait(entry)
        except Full:
            # Queue voll: nur Disk, kein Qdrant
            self._write_to_disk(entry)

    def suche(self, frage: str, limit: int = 5) -> List[Dict]:
        """Assoziative Suche ueber alle 3 Qdrant-Collections.

        Gibt die aehnlichsten Saetze zurueck, sortiert nach Score.
        """
        vector = self._embed(frage)
        if vector is None or not self._ensure_qdrant():
            return []

        results = []
        for coll in [COLLECTION_GEDANKEN, COLLECTION_MUSTER,
                     COLLECTION_EMERGENTIS]:
            try:
                hits = self._qdrant.query_points(
                    collection_name=coll,
                    query=vector,
                    limit=limit,
                )
                for pt in hits.points:
                    score = pt.score if hasattr(pt, "score") else 0.0
                    if score < 0.3:
                        continue
                    results.append({
                        "satz": pt.payload.get("satz", ""),
                        "score": round(score, 4),
                        "timestamp": pt.payload.get("timestamp", ""),
                        "schicht": pt.payload.get("schicht", ""),
                        "collection": coll,
                    })
            except Exception as e:
                logger.debug(f"[MS] Suche in {coll} fehlgeschlagen: {e}")

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def muster(self, zeitraum_tage: int = 7) -> List[Dict]:
        """Wiederkehrende Muster der letzten N Tage (Frequenzanalyse)."""
        verb_obj_count: Dict[str, int] = {}

        for days_back in range(zeitraum_tage):
            date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            log_path = os.path.join(GEDANKEN_DIR, f"{date}.log")
            if not os.path.exists(log_path):
                continue
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # Timestamp-Prefix entfernen: "2026-02-27T14:30:00 [SEHE] ..."
                        parts = line.split(" ", 1)
                        satz_part = (parts[1] if len(parts) > 1
                                     and "T" in parts[0] else line)
                        parsed = self.parse(satz_part)
                        if parsed:
                            key = f"{parsed['verb']}:{parsed['objekt']}"
                            verb_obj_count[key] = verb_obj_count.get(key, 0) + 1
            except Exception:
                continue

        # Nur wiederkehrende (>= 3x) zurueckgeben
        result = []
        for key, count in verb_obj_count.items():
            if count >= 3:
                verb, objekt = key.split(":", 1)
                result.append({
                    "verb": verb,
                    "objekt": objekt,
                    "count": count,
                    "schicht": self.schicht(verb),
                })
        result.sort(key=lambda x: x["count"], reverse=True)
        return result

    def schicht(self, verb: str) -> str:
        """Schicht-Zuordnung: guardian|shadow|emergentis|gemeinsam"""
        verb = verb.upper()
        if verb in SCHICHT_GUARDIAN:
            return "guardian"
        if verb in SCHICHT_SHADOW:
            return "shadow"
        if verb in SCHICHT_EMERGENTIS:
            return "emergentis"
        return "gemeinsam"

    def kontext(self, umgebung: str):
        """Kontext-Filter setzen und persistent speichern."""
        self._kontext = umgebung
        self._save_kontext(umgebung)
        logger.info(f"[MS] Kontext: {umgebung}")

    def vibe(self) -> float:
        """Aktueller Vibe-Score (-1.0 Shadow bis +1.0 Guardian)."""
        try:
            from core.core_integrator import get_core_integrator
            return get_core_integrator().get_dominance()
        except Exception:
            return 0.0

    # =====================================================================
    # Retention (30-Tage Cleanup fuer moloch_gedanken)
    # =====================================================================

    def tick_retention(self):
        """1x pro Aufruf den Counter hochzaehlen, 1x/Stunde aufraeumen.

        Gedacht fuer den 1Hz Status-Loop im Service.
        """
        self._cleanup_counter += 1
        if self._cleanup_counter < 3600:
            return
        self._cleanup_counter = 0
        threading.Thread(target=self._cleanup_gedanken, daemon=True,
                         name="MS-Cleanup").start()

    def _cleanup_gedanken(self):
        """Alte Eintraege aus moloch_gedanken loeschen (>30 Tage)."""
        if not self._ensure_qdrant():
            return
        try:
            from qdrant_client.models import Filter, FieldCondition, Range
            cutoff = (datetime.now() - timedelta(days=RETENTION_DAYS)).isoformat()
            self._qdrant.delete(
                collection_name=COLLECTION_GEDANKEN,
                points_selector=Filter(
                    must=[FieldCondition(
                        key="timestamp",
                        range=Range(lt=cutoff),
                    )]
                ),
            )
            logger.info(f"[MS] Retention: Eintraege vor {cutoff[:10]} bereinigt")
        except Exception as e:
            logger.warning(f"[MS] Retention fehlgeschlagen: {e}")

    # =====================================================================
    # Auto-Metadaten
    # =====================================================================

    def _auto_metadata(self, verb: str) -> Dict[str, Any]:
        """Automatische Metadaten: schicht, zone, vibe."""
        meta: Dict[str, Any] = {"schicht": self.schicht(verb)}
        try:
            from core.core_integrator import get_core_integrator
            ci = get_core_integrator()
            meta["zone"] = ci.get_personality_zone()
            meta["vibe"] = round(ci.get_dominance(), 3)
        except Exception:
            meta["zone"] = "unknown"
            meta["vibe"] = 0.0
        return meta

    # =====================================================================
    # Writer-Thread
    # =====================================================================

    def _writer_loop(self):
        """Background-Thread: Queue leeren, Disk + Qdrant schreiben."""
        while self._running:
            try:
                entry = self._queue.get(timeout=1.0)
            except Exception:
                continue
            self._write_to_disk(entry)
            self._write_to_qdrant(entry)

    def _write_to_disk(self, entry: Dict):
        """Satz in Tages-Log schreiben (append)."""
        try:
            satz = entry.get("satz", "")
            ts = entry.get("timestamp", datetime.now().isoformat())
            parsed = entry.get("parsed", {})
            verb = parsed.get("verb", "RAW")

            # Emergentis-Verben in eigenes Verzeichnis
            if verb in VERBEN_EMERGENTIS:
                log_dir = EMERGENTIS_DIR
            else:
                log_dir = GEDANKEN_DIR

            date_str = ts[:10]  # YYYY-MM-DD
            log_path = os.path.join(log_dir, f"{date_str}.log")

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{ts} {satz}\n")
        except Exception as e:
            logger.error(f"[MS] Disk-Write fehlgeschlagen: {e}")

    def _write_to_qdrant(self, entry: Dict):
        """Satz als Vektor in passende Collection schreiben."""
        if not self._ensure_qdrant():
            return

        satz = entry.get("satz", "")
        parsed = entry.get("parsed", {})
        verb = parsed.get("verb", "RAW")

        vector = self._embed(satz)
        if vector is None:
            return

        # Emergentis-Verben in eigene Collection
        if verb in VERBEN_EMERGENTIS:
            collection = COLLECTION_EMERGENTIS
        else:
            collection = COLLECTION_GEDANKEN

        try:
            from qdrant_client.models import PointStruct
            point_id = str(uuid.uuid4())
            payload = {
                "satz": satz,
                "verb": verb,
                "objekt": parsed.get("objekt", ""),
                "schicht": parsed.get("schicht", self.schicht(verb)),
                "zone": parsed.get("zone", ""),
                "vibe": parsed.get("vibe", 0.0),
                "kontext": entry.get("kontext", ""),
                "timestamp": entry.get("timestamp", ""),
            }
            self._qdrant.upsert(
                collection_name=collection,
                points=[PointStruct(id=point_id, vector=vector,
                                    payload=payload)],
            )
        except Exception as e:
            logger.debug(f"[MS] Qdrant-Write fehlgeschlagen: {e}")

    # =====================================================================
    # Kontext Persistence
    # =====================================================================

    def _load_kontext(self) -> str:
        """Kontext aus config/kontext.json laden."""
        try:
            if os.path.exists(KONTEXT_PATH):
                with open(KONTEXT_PATH, "r", encoding="utf-8") as f:
                    return json.load(f).get("kontext", "zuhause")
        except Exception:
            pass
        return "zuhause"

    def _save_kontext(self, umgebung: str):
        """Kontext persistent speichern (atomic write)."""
        try:
            data = {"kontext": umgebung,
                    "updated": datetime.now().isoformat()}
            tmp = KONTEXT_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, KONTEXT_PATH)
        except Exception as e:
            logger.error(f"[MS] Kontext speichern fehlgeschlagen: {e}")

    # =====================================================================
    # Muster Persistence
    # =====================================================================

    def save_muster(self, muster_list: List[Dict]):
        """Muster persistent speichern + in Qdrant."""
        # Disk (atomic write)
        try:
            tmp = MUSTER_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(muster_list, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, MUSTER_PATH)
        except Exception as e:
            logger.error(f"[MS] Muster speichern fehlgeschlagen: {e}")

        # Qdrant (best-effort)
        if not self._ensure_qdrant():
            return
        for m in muster_list:
            text = f"{m.get('verb', '')} {m.get('objekt', '')} x{m.get('count', 0)}"
            vector = self._embed(text)
            if vector is None:
                continue
            try:
                from qdrant_client.models import PointStruct
                # Deterministisches UUID pro Muster (Update statt Insert)
                pid = str(uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"muster.{m.get('verb','')}.{m.get('objekt','')}"
                ))
                self._qdrant.upsert(
                    collection_name=COLLECTION_MUSTER,
                    points=[PointStruct(id=pid, vector=vector, payload=m)],
                )
            except Exception:
                pass

    def _load_muster(self) -> List[Dict]:
        """Gespeicherte Muster laden."""
        try:
            if os.path.exists(MUSTER_PATH):
                with open(MUSTER_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return []


# =========================================================================
# Singleton
# =========================================================================
_instance: Optional[MolochSprache] = None
_instance_lock = threading.Lock()


def get_sprache() -> MolochSprache:
    """Singleton-Zugriff auf MolochSprache."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = MolochSprache()
    return _instance
