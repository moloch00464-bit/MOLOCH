"""Skill-Match-Tests fuer prompt_builder.

Verifiziert dass User-Prompts die erwarteten Skills triggern.
"""
import sys

from prompt_builder import match_skills

CASES = [
    ("schreib einen vision_auditor stub mit collect()", ["audit-pattern"]),
    ("post an PC_TO_PI mailbox mit topic discuss_x", ["mailbox-protocol"]),
    ("gstreamer pipeline mit hailo uint8 inferenz", ["gstreamer-hailo"]),
    ("registriere eine ipc action via register_action", ["ipc-pattern"]),
    ("speichere state nach /dev/shm atomic mit tempfile",
     ["atomic-write"]),
]


def main() -> int:
    fail = 0
    for prompt, expected in CASES:
        got = sorted(match_skills(prompt))
        exp = sorted(expected)
        ok = set(exp).issubset(set(got))
        status = "PASS" if ok else "FAIL"
        if not ok:
            fail += 1
        print(f"[{status}] expected {exp} got {got} -- '{prompt[:60]}'")
    print(f"\nResult: {len(CASES) - fail}/{len(CASES)} passed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
