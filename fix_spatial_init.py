#!/usr/bin/env python3
"""Fix: Spatial Learning Init."""

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

old = '''        self._spatial_learning = None  # Spatial Learning
        try:
            from core.perception_engine import PerceptionEngine
            from core.personality.personality_engine import get_personality_engine
            _pe = get_personality_engine()
            self._perception = PerceptionEngine(personality_engine=_pe)'''

new = '''        self._spatial_learning = None  # Spatial Learning
        try:
            from core.perception_engine import PerceptionEngine
            from core.personality.personality_engine import get_personality_engine
            _pe = get_personality_engine()
            self._perception = PerceptionEngine(personality_engine=_pe)
            self._spatial_learning = get_spatial_learning()  # Init Spatial Learning'''

if old in code:
    code = code.replace(old, new)
    print('FIX: Init SpatialLearning with PerceptionEngine - OK')
    with open(svc, 'w') as f:
        f.write(code)
    compile(open(svc).read(), svc, 'exec')
    print('Syntax OK.')
else:
    print('ANCHOR NOT FOUND!')
    import sys
    sys.exit(1)
