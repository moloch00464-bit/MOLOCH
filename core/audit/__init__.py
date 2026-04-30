"""Audit-Orchestrator (Welle 8).

Aggregiert Pi + PC + Persona + Mailbox Layer in /dev/shm/audit_state.json.
Subprocess-Tick alle 60s (KEIN dauerhafter Daemon — Pi 4 GB RAM).

Schema siehe core/audit/audit_orchestrator.py docstring + .claude/agents/audit.md.
"""
