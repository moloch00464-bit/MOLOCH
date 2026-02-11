#!/usr/bin/env python3
"""
M.O.L.O.C.H. Test Configuration
================================

Shared fixtures and markers for pytest.
"""

import sys
import os
import pytest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "requires_qdrant: test needs running Qdrant instance")
    config.addinivalue_line("markers", "requires_hailo: test needs Hailo NPU hardware")
    config.addinivalue_line("markers", "requires_camera: test needs camera connection")
    config.addinivalue_line("markers", "requires_mic: test needs microphone")
    config.addinivalue_line("markers", "requires_tts: test needs Piper TTS")
    config.addinivalue_line("markers", "requires_claude: test needs Claude API key")
    config.addinivalue_line("markers", "slow: test takes >5 seconds")


@pytest.fixture
def tmp_memory_dir(tmp_path):
    """Temporary directory for memory files."""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    return mem_dir
