"""MOLOCH Coder-Tentakel — Skill-Routing + Ollama-Call.

Liest User-Prompt, matched Skills basierend auf Trigger-Keywords,
injiziert vor Prompt, sendet an moloch-coder.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import requests

SKILLS_DIR = Path(__file__).parent / "skills"

SKILL_TRIGGERS = {
    "audit-pattern": [
        "auditor", "collect", "audit_state", "score", "merge_component",
    ],
    "mailbox-protocol": [
        "mailbox", "PC_TO_PI", "PI_TO_PC", "topic", "/mailbox",
    ],
    "gstreamer-hailo": [
        "gstreamer", "pipeline", "hailo", "uint8", "float32",
        "SCRFD", "ArcFace", "ReID", "perception",
    ],
    "ipc-pattern": [
        "ipc", "moloch_service", "register_action", "route_action",
        "spotify_play", "shell=True",
    ],
    "atomic-write": [
        "atomic", "/dev/shm", "tempfile", "os.replace",
        "race-condition", "audit_state.json",
    ],
}

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "moloch-coder"
DEFAULT_TIMEOUT = 120


def match_skills(user_prompt: str) -> List[str]:
    p = user_prompt.lower()
    return [
        skill for skill, kws in SKILL_TRIGGERS.items()
        if any(k.lower() in p for k in kws)
    ]


def build_prompt(user_prompt: str) -> str:
    matched = match_skills(user_prompt)
    blocks = []
    for skill in matched:
        skill_path = SKILLS_DIR / f"{skill}.md"
        if skill_path.exists():
            blocks.append(
                f"## Skill: {skill}\n\n"
                + skill_path.read_text(encoding="utf-8")
            )
    if blocks:
        prefix = "\n\n---\n\n".join(blocks) + "\n\n---\n\n"
    else:
        prefix = ""
    return prefix + user_prompt


def call_coder(user_prompt: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    payload = {
        "model": MODEL,
        "prompt": build_prompt(user_prompt),
        "stream": False,
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "")


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("usage: prompt_builder.py <user-prompt>", file=sys.stderr)
        return 1
    user_prompt = " ".join(argv[1:])
    matched = match_skills(user_prompt)
    print(f"# matched skills: {matched or '(none)'}", file=sys.stderr)
    print(call_coder(user_prompt))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
