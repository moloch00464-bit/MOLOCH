#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Speech Module
==========================

Speech-to-Text using Hailo-10H NPU acceleration.
"""

from .hailo_whisper import HailoWhisper, get_whisper

__all__ = ['HailoWhisper', 'get_whisper']
