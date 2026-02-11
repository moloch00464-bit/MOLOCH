#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Persistent Memory
================================

Speichert Fakten und Konversationskontext persistent auf Disk.
Wird beim Start geladen und in den System-Prompt injiziert.

Storage:
  ~/moloch/data/memory/user_knowledge.json  - Fakten ueber Benutzer
  ~/moloch/data/memory/conversation_log.json - Letzte N Konversationsturns

Mechanismus:
  Claude bekommt Instruktion, [REMEMBER: key=value] Tags zu schreiben
  wenn er etwas Wichtiges lernt (Spitznamen, Vorlieben, Fakten).
  Diese Tags werden geparst und persistent gespeichert.
  Beim naechsten Start werden sie in den System-Prompt injiziert.
"""

import json
import os
import re
import time
import logging
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MEMORY_DIR = os.path.expanduser("~/moloch/data/memory")
KNOWLEDGE_FILE = os.path.join(MEMORY_DIR, "user_knowledge.json")
CONVERSATION_FILE = os.path.join(MEMORY_DIR, "conversation_log.json")

MAX_CONVERSATION_TURNS = 20  # Letzte 20 Turns persistent
MAX_PROMPT_TURNS = 10        # Davon 10 in den Prompt


class PersistentMemory:
    """
    Persistentes Gedaechtnis fuer M.O.L.O.C.H.

    Zwei Speicher:
    1. knowledge: Key-Value Fakten (Spitznamen, Vorlieben, etc.)
    2. conversation_log: Letzte N Konversationsturns

    Beide werden auf Disk gespeichert und beim Start geladen.
    """

    def __init__(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        self._lock = threading.Lock()
        self.knowledge: Dict[str, str] = {}
        self.conversation_log: List[Dict] = []
        self._load()

    def _load(self):
        """Lade Gedaechtnis von Disk."""
        # Knowledge
        if os.path.exists(KNOWLEDGE_FILE):
            try:
                with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                    self.knowledge = json.load(f)
                logger.info(f"[MEMORY] {len(self.knowledge)} Fakten geladen")
            except Exception as e:
                logger.error(f"[MEMORY] Knowledge load error: {e}")
                self.knowledge = {}

        # Conversation log
        if os.path.exists(CONVERSATION_FILE):
            try:
                with open(CONVERSATION_FILE, "r", encoding="utf-8") as f:
                    self.conversation_log = json.load(f)
                logger.info(f"[MEMORY] {len(self.conversation_log)} Konversations-Turns geladen")
            except Exception as e:
                logger.error(f"[MEMORY] Conversation load error: {e}")
                self.conversation_log = []

    def _save_knowledge(self):
        """Speichere Wissen auf Disk."""
        try:
            with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.knowledge, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[MEMORY] Knowledge save error: {e}")

    def _save_conversation(self):
        """Speichere Konversationslog auf Disk."""
        try:
            # Nur letzte N Turns behalten
            if len(self.conversation_log) > MAX_CONVERSATION_TURNS:
                self.conversation_log = self.conversation_log[-MAX_CONVERSATION_TURNS:]
            with open(CONVERSATION_FILE, "w", encoding="utf-8") as f:
                json.dump(self.conversation_log, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[MEMORY] Conversation save error: {e}")

    # === WRITE ===

    def remember(self, key: str, value: str):
        """Speichere einen Fakt persistent."""
        with self._lock:
            self.knowledge[key] = value
            self._save_knowledge()
        logger.info(f"[MEMORY] Gemerkt: {key} = {value}")
        # Auch in Qdrant Vektorspeicher
        try:
            from core.memory.vector_memory import get_vector_memory
            vm = get_vector_memory()
            vm.store(f"{key}: {value}", category="fact", key=key)
        except Exception as e:
            logger.debug(f"[MEMORY] Vector store (non-critical): {e}")

    def forget(self, key: str):
        """Vergesse einen Fakt."""
        with self._lock:
            removed = self.knowledge.pop(key, None)
            if removed:
                self._save_knowledge()
                logger.info(f"[MEMORY] Vergessen: {key}")
                # Auch aus Qdrant loeschen
                try:
                    from core.memory.vector_memory import get_vector_memory
                    vm = get_vector_memory()
                    vm.delete(key)
                except Exception:
                    pass

    def add_turn(self, role: str, content: str):
        """Fuege eine Konversationsrunde hinzu und speichere."""
        with self._lock:
            self.conversation_log.append({
                "role": role,
                "content": content[:500],  # Max 500 Zeichen pro Turn
                "timestamp": time.time()
            })
            self._save_conversation()

    def extract_memories(self, text: str) -> str:
        """
        Parse [REMEMBER: key=value] Tags aus Text und speichere sie.

        Returns:
            Text ohne die [REMEMBER:] Tags (fuer Anzeige/TTS)
        """
        pattern = r'\[REMEMBER:\s*(.+?)\s*=\s*(.+?)\s*\]'
        matches = re.findall(pattern, text)
        for key, value in matches:
            self.remember(key.strip(), value.strip())

        # Tags aus dem Text entfernen (User/TTS soll sie nicht sehen)
        cleaned = re.sub(pattern, '', text).strip()
        # Doppelte Leerzeichen/Newlines aufraumen
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
        return cleaned

    # === READ ===

    def get_knowledge(self) -> Dict[str, str]:
        """Alle gespeicherten Fakten."""
        return dict(self.knowledge)

    def get_recent_conversation(self, n: int = None) -> List[Dict]:
        """Letzte N Konversationsturns."""
        n = n or MAX_PROMPT_TURNS
        return self.conversation_log[-n:]

    # === PROMPT INJECTION ===

    def to_prompt_section(self) -> str:
        """
        Generiere System-Prompt-Abschnitt aus Gedaechtnis.

        Wird in den System-Prompt injiziert damit Claude
        persistent gespeichertes Wissen hat.
        """
        sections = []

        if self.knowledge:
            sections.append("=== LANGZEITGEDAECHTNIS ===")
            sections.append("Dinge die du dir gemerkt hast (persistent gespeichert):")
            for key, value in self.knowledge.items():
                sections.append(f"- {key}: {value}")

        recent = self.get_recent_conversation(MAX_PROMPT_TURNS)
        if recent:
            sections.append("")
            sections.append("=== LETZTE KONVERSATION (vor Neustart) ===")
            for turn in recent:
                role_label = "Markus" if turn["role"] == "user" else "Du"
                content = turn["content"]
                if len(content) > 200:
                    content = content[:200] + "..."
                sections.append(f"  {role_label}: {content}")

        return "\n".join(sections) if sections else ""

    def get_memory_instruction(self) -> str:
        """
        Instruktionen fuer Claude, wie er das Gedaechtnis nutzen soll.
        Wird einmalig in den System-Prompt eingefuegt.
        """
        return """
=== GEDAECHTNIS-SYSTEM ===
Du hast ein LANGZEITGEDAECHTNIS das Neustarts ueberlebt.
Wenn du etwas Wichtiges ueber eine Person lernst, speichere es mit:
[REMEMBER: schluessel=wert]

Wann speichern:
- Spitznamen: [REMEMBER: Markus_Spitzname=PIGH0ST]
- Vorlieben: [REMEMBER: Markus_Lieblingsmusik=Dark Wave]
- Wichtige Fakten: [REMEMBER: Markus_Projekt=M.O.L.O.C.H. Hauskobold]
- Neue Personen: [REMEMBER: Person_Tom=Nachbar von Markus]
- Abmachungen: [REMEMBER: Versprechen_1=Markus will morgen frueh aufstehen]

Regeln:
- NUR fuer dauerhafte, wichtige Fakten (nicht fuer Smalltalk)
- Der [REMEMBER:] Tag wird automatisch entfernt bevor der User ihn sieht
- Du kannst mehrere [REMEMBER:] Tags in einer Antwort verwenden
- Schluessel mit Unterstrich statt Leerzeichen
- SPARSAM benutzen - nur echte Fakten, kein Muell"""

    def sync_to_vector(self):
        """Sync alle JSON-Fakten nach Qdrant (fuer Background-Thread)."""
        try:
            from core.memory.vector_memory import get_vector_memory
            vm = get_vector_memory()
            vm.sync_knowledge(self.knowledge)
        except Exception as e:
            logger.debug(f"[MEMORY] Vector sync: {e}")

    def __str__(self):
        return f"PersistentMemory(knowledge={len(self.knowledge)}, turns={len(self.conversation_log)})"


# Singleton
_memory_instance: Optional[PersistentMemory] = None
_memory_lock = threading.Lock()


def get_memory() -> PersistentMemory:
    """Get or create PersistentMemory singleton."""
    global _memory_instance
    if _memory_instance is None:
        with _memory_lock:
            if _memory_instance is None:
                _memory_instance = PersistentMemory()
    return _memory_instance
