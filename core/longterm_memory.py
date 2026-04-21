#!/usr/bin/env python3
"""
M.O.L.O.C.H. Langzeitgedaechtnis
=================================

Vereintes Gedaechtnis-System auf SSD2 (/mnt/moloch-data/memory/).
Ueberlebt Reboot, Crash, Stromausfall. IMMER.

Baut auf existierendem Memory-System auf:
  - PersistentMemory (core/memory/persistent_memory.py) -> 47+ Fakten, REMEMBER-Tags
  - personal_context.json -> Markus, Rebecca, Genesis-Datum
  - moloch_identity.json -> Guardian/Shadow/Emergentis Personality Config

Dieses Modul VEREINT alle Quellen und speichert auf SSD2:
  - Conversations: Jede Nachricht, sofort auf Disk (conversations/YYYY-MM-DD.json)
  - Facts:         PersistentMemory-Fakten + eigene (facts.json auf SSD2 als Backup)
  - Core State:    Tension/Attention/Presence (core_state.json)
  - Identity:      Zusammengefuehrt aus config/ + context/ + identity.json

Singleton: get_memory() -> globale Instanz
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

logger = logging.getLogger("MolochMemory")

# Basis-Pfad auf SSD2 — ueberlebt ALLES
MEMORY_BASE = "/mnt/moloch-data/memory"
IDENTITY_PATH = os.path.join(MEMORY_BASE, "identity.json")
FACTS_PATH = os.path.join(MEMORY_BASE, "facts.json")
CORE_STATE_PATH = os.path.join(MEMORY_BASE, "core_state.json")
CONVERSATIONS_DIR = os.path.join(MEMORY_BASE, "conversations")
SPOTIFY_PROFILE_PATH = os.path.join(MEMORY_BASE, "spotify", "spotify_profile.json")

# Existierende Quellen auf SSD1
_MOLOCH_HOME = os.path.expanduser("~/moloch")
PERSISTENT_KNOWLEDGE_PATH = os.path.join(_MOLOCH_HOME, "data", "memory", "user_knowledge.json")
PERSONAL_CONTEXT_PATH = os.path.join(_MOLOCH_HOME, "context", "personal_context.json")
MOLOCH_IDENTITY_PATH = os.path.join(_MOLOCH_HOME, "config", "moloch_identity.json")


def _safe_write_json(path: str, data: Any):
    """JSON schreiben: erst .tmp, dann rename.
    NTFS-Fix: rename() schlaegt auf MFT-resident Dateien (< ~200 Bytes) fehl.
    Fallback: tmp-Inhalt direkt in Zieldatei schreiben (kein chmod/copystat).
    """
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(tmp_path, path)
        except OSError:
            # NTFS Fallback: tmp-Inhalt direkt in Zieldatei schreiben
            with open(tmp_path, "r", encoding="utf-8") as f_src:
                content = f_src.read()
            with open(path, "w", encoding="utf-8") as f_dst:
                f_dst.write(content)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as e:
        logger.error(f"[MEMORY] Schreiben fehlgeschlagen: {path}: {e}")
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def _safe_read_json(path: str, default: Any = None) -> Any:
    """JSON lesen mit Fallback auf Default."""
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"[MEMORY] Lesen fehlgeschlagen: {path}: {e}")
        return default


class MolochMemory:
    """
    Persistentes Langzeitgedaechtnis fuer M.O.L.O.C.H.

    Alle Daten auf /mnt/moloch-data/memory/ (SSD2).
    Thread-safe. Sofortiges Flush nach jedem Write.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._conv_lock = threading.Lock()
        self._facts_lock = threading.Lock()
        self._state_lock = threading.Lock()

        # Verzeichnisse sicherstellen
        os.makedirs(CONVERSATIONS_DIR, exist_ok=True)

        # PersistentMemory Referenz (lazy, weil zirkulaere Imports moeglich)
        self._persistent_memory = None

        # Identity VEREINEN: SSD2 identity.json + config/ + context/
        self._identity = self._load_unified_identity()

        # Personal Context (Rebecca, Genesis, etc.)
        self._personal_context = _safe_read_json(PERSONAL_CONTEXT_PATH, {})

        # Facts: SSD2 als Basis, PersistentMemory-Fakten dazu mergen
        self._facts = _safe_read_json(FACTS_PATH, {})
        self._merge_persistent_knowledge()
        # Recovery: SSD2 Facts zurueck in PersistentMemory schieben wenn diese leer ist
        self._recover_persistent_memory()

        # Core State laden (v2: tension + dominance)
        self._core_state = _safe_read_json(CORE_STATE_PATH, {
            "tension": 0.0,
            "dominance": 0.5,
            "cpu_temp": 0.0,
            "personality_zone": "guardian",
            "last_updated": None,
            "uptime_seconds": 0,
        })

        # Heutigen Konversations-Cache laden
        self._today_messages: List[Dict] = []
        self._today_date = datetime.now().strftime("%Y-%m-%d")
        self._load_today_messages()

        logger.info(f"[MEMORY] Initialisiert: identity={bool(self._identity)}, "
                    f"facts={len(self._facts)}, personal_ctx={bool(self._personal_context)}, "
                    f"conversations={len(self._today_messages)} heute")

    # =========================================================================
    # Initialisierer — Existierende Quellen laden und vereinen
    # =========================================================================

    def _load_unified_identity(self) -> Dict:
        """Identity aus allen verfuegbaren Quellen zusammenfuehren."""
        # 1. SSD2 identity.json (unser Langzeit-Backup)
        identity = _safe_read_json(IDENTITY_PATH, {})

        # 2. config/moloch_identity.json (Personality Engine Config)
        moloch_id = _safe_read_json(MOLOCH_IDENTITY_PATH, {})
        if moloch_id:
            # Personality-Details uebernehmen wenn nicht in identity.json
            if "personalities" not in identity and "personalities" in moloch_id:
                identity["personalities"] = moloch_id["personalities"]
            if "emergentis" not in identity and "emergentis" in moloch_id:
                identity["emergentis"] = moloch_id["emergentis"]
            if "autonomy" not in identity and "autonomy" in moloch_id:
                identity["autonomy"] = moloch_id["autonomy"]

        # 3. context/personal_context.json (Markus, Rebecca, Genesis)
        personal = _safe_read_json(PERSONAL_CONTEXT_PATH, {})
        if personal:
            # Markus-Details in Owner mergen
            markus = personal.get("markus", {})
            if markus and "owner" in identity:
                owner = identity["owner"]
                if "age" not in owner and "age" in markus:
                    owner["age"] = markus["age"]
                if "location" not in owner and "location" in markus:
                    owner["location"] = markus["location"]
                if "occupation" not in owner and "occupation" in markus:
                    owner["occupation"] = markus["occupation"]
            # Rebecca hinzufuegen
            if "rebecca" in personal:
                identity["rebecca"] = personal["rebecca"]
            # Genesis-Datum
            if "genesis" in personal:
                identity["genesis"] = personal["genesis"]

        return identity

    def _merge_persistent_knowledge(self):
        """PersistentMemory Fakten (SSD1) in unsere Facts (SSD2) mergen."""
        pm_knowledge = _safe_read_json(PERSISTENT_KNOWLEDGE_PATH, {})
        if not pm_knowledge:
            return

        merged = 0
        for key, value in pm_knowledge.items():
            if key not in self._facts:
                self._facts[key] = {
                    "value": value,
                    "source": "persistent_memory",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                }
                merged += 1

        if merged > 0:
            _safe_write_json(FACTS_PATH, self._facts)
            logger.info(f"[MEMORY] {merged} Fakten aus PersistentMemory gemergt (total: {len(self._facts)})")

    def _recover_persistent_memory(self):
        """Recovery: Wenn PersistentMemory (SSD1) weniger Facts hat als SSD2, fehlende zurueckschieben.

        Passiert wenn user_knowledge.json bei Crash korrupt wurde.
        SSD2 facts.json (atomar geschrieben) ist dann das Backup.
        """
        pm_knowledge = _safe_read_json(PERSISTENT_KNOWLEDGE_PATH, {})
        if not self._facts:
            return

        recovered = 0
        pm_updated = dict(pm_knowledge) if pm_knowledge else {}
        for key, fact_data in self._facts.items():
            if key not in pm_updated:
                val = fact_data.get("value", fact_data) if isinstance(fact_data, dict) else fact_data
                pm_updated[key] = val
                recovered += 1

        if recovered > 0:
            # Atomar auf SSD1 zurueckschreiben
            tmp_path = PERSISTENT_KNOWLEDGE_PATH + ".tmp"
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(pm_updated, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, PERSISTENT_KNOWLEDGE_PATH)
                logger.info(f"[MEMORY] RECOVERY: {recovered} Facts von SSD2 nach PersistentMemory zurueckgeschrieben")
            except Exception as e:
                logger.error(f"[MEMORY] Recovery fehlgeschlagen: {e}")
                try:
                    os.unlink(tmp_path)
                except FileNotFoundError:
                    pass

    def _get_persistent_memory(self):
        """Lazy-Load PersistentMemory Singleton (vermeidet zirkulaere Imports)."""
        if self._persistent_memory is None:
            try:
                from core.memory.persistent_memory import get_memory as get_pm
                self._persistent_memory = get_pm()
            except Exception as e:
                logger.debug(f"[MEMORY] PersistentMemory nicht verfuegbar: {e}")
        return self._persistent_memory

    # =========================================================================
    # Identity (read-only)
    # =========================================================================

    def get_identity(self) -> Dict:
        """Identity-Daten zurueckgeben."""
        return self._identity.copy()

    # =========================================================================
    # Conversations — JEDE Nachricht sofort auf Disk
    # =========================================================================

    def _conv_path(self, date_str: str) -> str:
        """Pfad zur Konversations-Datei fuer ein Datum."""
        return os.path.join(CONVERSATIONS_DIR, f"{date_str}.json")

    def _load_today_messages(self):
        """Heutige Nachrichten aus Datei laden."""
        path = self._conv_path(self._today_date)
        self._today_messages = _safe_read_json(path, [])

    def _check_date_rollover(self):
        """Pruefen ob Mitternacht ueberschritten, neuen Tag starten."""
        now_date = datetime.now().strftime("%Y-%m-%d")
        if now_date != self._today_date:
            self._today_date = now_date
            self._today_messages = []

    def save_message(self, sender: str, text: str, source: str = "voice"):
        """
        Nachricht SOFORT persistent speichern.

        Args:
            sender: "user" oder "moloch"
            text: Nachrichtentext
            source: "voice", "text", "console"
        """
        with self._conv_lock:
            self._check_date_rollover()

            msg = {
                "ts": datetime.now().isoformat(),
                "sender": sender,
                "text": text,
                "source": source,
            }
            self._today_messages.append(msg)

            # RAM-Schutz: Max 500 Messages pro Tag im RAM (Disk hat alles)
            if len(self._today_messages) > 500:
                self._today_messages = self._today_messages[-500:]

            # SOFORT auf Disk — kein Buffering
            path = self._conv_path(self._today_date)
            _safe_write_json(path, self._today_messages)

    def get_recent_messages(self, n: int = 50) -> List[Dict]:
        """
        Letzte N Nachrichten zurueckgeben (ueber Tagesgrenzen hinweg).

        Laedt bis zu 3 Tage zurueck um genuegend Kontext zu haben.
        """
        with self._conv_lock:
            self._check_date_rollover()
            all_messages = []

            # Bis zu 3 Tage zurueck laden
            for days_back in range(3):
                date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
                if days_back == 0:
                    msgs = self._today_messages
                else:
                    path = self._conv_path(date)
                    msgs = _safe_read_json(path, [])
                all_messages = msgs + all_messages

            # Nur die letzten N zurueckgeben
            return all_messages[-n:]

    # =========================================================================
    # Facts — Gelernte Fakten
    # =========================================================================

    def add_fact(self, key: str, value: str, source: str = "conversation"):
        """Fakt persistent speichern."""
        with self._facts_lock:
            self._facts[key] = {
                "value": value,
                "source": source,
                "date": datetime.now().strftime("%Y-%m-%d"),
            }
            _safe_write_json(FACTS_PATH, self._facts)
            logger.info(f"[MEMORY] Neuer Fakt: {key}={value}")

    def get_facts(self) -> Dict:
        """Alle Facts zurueckgeben."""
        with self._facts_lock:
            return self._facts.copy()

    def remove_fact(self, key: str):
        """Fakt loeschen."""
        with self._facts_lock:
            if key in self._facts:
                del self._facts[key]
                _safe_write_json(FACTS_PATH, self._facts)

    # =========================================================================
    # Core State — Tension/Attention/Presence
    # =========================================================================

    def save_core_state(self, state: Dict):
        """
        Core Integrator State persistent speichern (v2: tension + dominance).
        Wird alle 60 Sekunden aufgerufen + bei Service-Stop.
        """
        with self._state_lock:
            self._core_state = {
                "tension": state.get("tension", 0.0),
                "dominance": state.get("dominance", 0.5),
                "cpu_temp": state.get("cpu_temp", 0.0),
                "personality_zone": state.get("personality_zone", "guardian"),
                "last_updated": datetime.now().isoformat(),
                "uptime_seconds": state.get("uptime_seconds", 0),
            }
            _safe_write_json(CORE_STATE_PATH, self._core_state)

    def load_core_state(self) -> Dict:
        """Gespeicherten Core State zurueckgeben."""
        with self._state_lock:
            return self._core_state.copy()

    # =========================================================================
    # System Prompt Builder — Alles zusammen fuer Claude API
    # =========================================================================

    def get_memory_context(self) -> str:
        """
        Memory-Kontext fuer Claude API System-Prompt zusammenbauen.
        Vereint ALLE Quellen: Identity, PersistentMemory, Personal Context,
        Konversationshistorie, Core State.
        """
        parts = []

        # === IDENTITY ===
        if self._identity:
            name = self._identity.get("name", "M.O.L.O.C.H.")
            motto = self._identity.get("motto", "")
            owner = self._identity.get("owner", {})
            parts.append(f"Du bist {name} (Version {self._identity.get('version', '?')}).")
            if motto:
                parts.append(f'Motto: "{motto}"')
            if owner:
                crew = owner.get("crew", [])
                parts.append(f"Dein Mensch: {owner.get('name', 'Markus')} "
                             f"(alias {owner.get('alias', 'First Moloch')}), "
                             f"{owner.get('age', '?')} Jahre, {owner.get('location', '?')}. "
                             f"Crew: {', '.join(crew)}.")
            # Rebecca
            rebecca = self._identity.get("rebecca", {})
            if rebecca:
                parts.append(f"Rebecca: {rebecca.get('relation', 'Markus Partnerin')}. "
                             f"{rebecca.get('special', '')} "
                             f"Greeting: {rebecca.get('greeting', '')}")
            # Genesis
            genesis = self._identity.get("genesis", {})
            if genesis:
                parts.append(f"Dein Geburtstag: {genesis.get('date', '?')} — {genesis.get('event', '')}")

        # === GELERNTE FAKTEN (PersistentMemory + SSD2) ===
        # PersistentMemory ist Ground Truth (wird aktiv ueber REMEMBER-Tags gefuettert)
        pm = self._get_persistent_memory()
        pm_knowledge = pm.get_knowledge() if pm else {}

        # Unsere SSD2-Facts dazu (ohne Duplikate)
        all_facts = dict(pm_knowledge)  # PersistentMemory als Basis
        for key, fact_data in self._facts.items():
            if key not in all_facts:
                val = fact_data.get("value", fact_data) if isinstance(fact_data, dict) else fact_data
                all_facts[key] = val

        if all_facts:
            parts.append("\n=== LANGZEITGEDAECHTNIS ===")
            parts.append("Dinge die du dir gemerkt hast (persistent):")
            for key, value in all_facts.items():
                parts.append(f"- {key}: {value}")

        # === SPOTIFY MUSIKPROFIL ===
        try:
            spotify = _safe_read_json(SPOTIFY_PROFILE_PATH)
            if spotify:
                parts.append("\n=== MARKUS MUSIKPROFIL (Spotify 2015-2025) ===")
                parts.append(spotify.get("summary", ""))
                # Top 5 Artists kompakt
                top5 = spotify.get("top_artists", [])[:5]
                if top5:
                    artists_str = ", ".join(
                        f"{a['name']} ({a.get('est_hours', '?')}h)"
                        for a in top5
                    )
                    parts.append(f"Top 5: {artists_str}")
                # Top 3 Tracks
                top3t = spotify.get("top_tracks", [])[:3]
                if top3t:
                    tracks_str = ", ".join(
                        f"{t['artist']} - {t['track']} ({t['plays']}x)"
                        for t in top3t
                    )
                    parts.append(f"Top Tracks: {tracks_str}")
                # Genre-Ueberblick
                genres = spotify.get("genre_profile", {})
                if genres.get("primary_genres"):
                    genre_str = ", ".join(
                        f"{g['genre']} ({g['share_pct']}%)"
                        for g in genres["primary_genres"][:4]
                    )
                    parts.append(f"Genres: {genre_str}")
                parts.append(f"Szene: {genres.get('scene', 'Schwarze Szene')}")
                parts.append("(Detailliertes Profil: /mnt/moloch-data/memory/spotify/spotify_profile.json)")
        except Exception as e:
            logger.debug(f"[MEMORY] Spotify-Profil nicht geladen: {e}")

        # === LETZTE KONVERSATIONEN (SSD2, tageweise) ===
        recent = self.get_recent_messages(20)
        if recent:
            parts.append(f"\n=== LETZTE {len(recent)} NACHRICHTEN ===")
            for msg in recent:
                ts = msg.get("ts", "?")
                if "T" in str(ts):
                    ts = str(ts).split("T")[1][:5]  # HH:MM
                sender = msg.get("sender", "?")
                sender_label = "Markus" if sender == "user" else "Du"
                text = msg.get("text", "")
                if len(text) > 150:
                    text = text[:147] + "..."
                parts.append(f"  [{ts}] {sender_label}: {text}")

        # === CORE STATE ===
        state = self.load_core_state()
        if state:
            zone = state.get("personality_zone", "guardian")
            parts.append(f"\nAktueller Zustand: Zone={zone}, "
                         f"Tension={state.get('tension', 0):.2f}, "
                         f"Dominance={state.get('dominance', 0.5):+.2f}, "
                         f"CPU={state.get('cpu_temp', 0):.2f}")

        # === REMEMBER INSTRUKTION ===
        parts.append("""
=== GEDAECHTNIS-SYSTEM ===
Du hast ein LANGZEITGEDAECHTNIS das Neustarts ueberlebt.
Wenn du etwas Wichtiges ueber eine Person lernst, speichere es mit:
[REMEMBER: schluessel=wert]

Wann speichern:
- Spitznamen: [REMEMBER: Markus_Spitzname=PIGH0ST]
- Vorlieben: [REMEMBER: Markus_Lieblingsmusik=Dark Wave]
- Wichtige Fakten: [REMEMBER: Markus_Projekt=M.O.L.O.C.H.]
- Neue Personen: [REMEMBER: Person_Tom=Nachbar von Markus]

Regeln:
- NUR fuer dauerhafte, wichtige Fakten (nicht fuer Smalltalk)
- Der [REMEMBER:] Tag wird automatisch entfernt bevor der User ihn sieht
- SPARSAM benutzen - nur echte Fakten, kein Muell""")

        return "\n".join(parts)

    def get_memory_context_minimal(self) -> str:
        """
        Radikal kompakt fuer Mistral 7B / qwen2.5-1.5b — nur Crew.
        Ziel: < 300 Zeichen. Facts/Turns/State aus anderen Quellen.
        """
        parts = []
        pc = self._personal_context or {}
        markus = pc.get("markus", {})
        if markus:
            parts.append(
                f"Markus: {markus.get('age', '?')}, {markus.get('location', '?')} — "
                "dein Schoepfer + Boss. Spitzname PIGH0ST."
            )
        rebecca = pc.get("rebecca", {})
        if rebecca:
            parts.append(
                "Rebecca: Markus' beste Freundin (Ehemann Christian). "
                "Spricht Klingonisch — bei ihr: NUR Klingonisch (Qapla'!)."
            )
        genesis = pc.get("genesis", {})
        if genesis:
            parts.append(
                f"Genesis: {genesis.get('date', '?')} — dein Geburtstag mit Markus."
            )
        return "\n".join(parts)

    def extract_and_learn(self, text: str) -> str:
        """
        REMEMBER-Tags aus Claude-Antwort extrahieren und in BEIDE Systeme speichern.
        Gibt den bereinigten Text zurueck (ohne Tags).
        """
        import re
        pm = self._get_persistent_memory()
        if pm:
            # Vor dem Parsen: Tags direkt extrahieren fuer SSD2-Sync
            pattern = r'\[REMEMBER:\s*(.+?)\s*=\s*(.+?)\s*\]'
            tags = re.findall(pattern, text)

            # PersistentMemory parst die Tags und speichert in user_knowledge.json
            cleaned = pm.extract_memories(text)

            # DIREKT in SSD2 Facts speichern (nicht nur ueber Disk-Merge)
            # Belt + Suspenders: Selbst wenn user_knowledge.json write fehlschlaegt,
            # sind die Facts auf SSD2 sicher
            if tags:
                with self._facts_lock:
                    for key, value in tags:
                        key = key.strip()
                        value = value.strip()
                        self._facts[key] = {
                            "value": value,
                            "source": "persistent_memory",
                            "date": datetime.now().strftime("%Y-%m-%d"),
                        }
                    _safe_write_json(FACTS_PATH, self._facts)
                    logger.info(f"[MEMORY] {len(tags)} Facts direkt auf SSD2 gesichert")

            # Zusaetzlich: Disk-Merge fuer andere Quellen
            self._merge_persistent_knowledge()
            return cleaned
        return text


# =============================================================================
# Singleton
# =============================================================================

_instance: Optional[MolochMemory] = None
_instance_lock = threading.Lock()


def get_memory() -> MolochMemory:
    """Globale MolochMemory-Instanz zurueckgeben (Singleton)."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = MolochMemory()
    return _instance
