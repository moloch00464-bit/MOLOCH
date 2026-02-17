#!/usr/bin/env python3
"""Fix: Model Configuration Race Condition

PROBLEM:
- Idle Pre-Load thread calls _configure_model("scrfd")
- Perception Engine thread calls _configure_model("pose") GLEICHZEITIG
- Beide laufen parallel -> 3-4 Modelle aktiv -> VIOLATION -> HailoRT timeout -> Service hung

FIX:
1. Add _configure_lock in __init__
2. _configure_model: acquire() at start, release() in finally
3. _unconfigure_model: same pattern

Result: Only ONE thread can configure/unconfigure at a time -> No more races
"""

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

fixes = 0

# FIX 1: Add _configure_lock in __init__
old_init = '''        # NPU Model swapping
        self._models = {}               # name -> InferModel
        self._active_ctx = {}           # name -> {ctx_mgr, configured, bindings, output_buffers}
        self._output_names = {}         # name -> List[str]
        self._ctx_lock = threading.Lock()'''

new_init = '''        # NPU Model swapping
        self._models = {}               # name -> InferModel
        self._active_ctx = {}           # name -> {ctx_mgr, configured, bindings, output_buffers}
        self._output_names = {}         # name -> List[str]
        self._ctx_lock = threading.Lock()
        self._configure_lock = threading.Lock()  # Serialize model configure/unconfigure (race protection)'''

if old_init in code:
    code = code.replace(old_init, new_init)
    print('FIX 1: Add _configure_lock - OK')
    fixes += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

# FIX 2: acquire() at start of _configure_model
old_start = '''    def _configure_model(self, name):
        """Konfiguriere Modell persistent (einmalig ~400ms, danach 0ms)."""
        if name in self._active_ctx:
            logger.info(f"[CONFIGURE] {name}: bereits konfiguriert, skip")
            return'''

new_start = '''    def _configure_model(self, name):
        """Konfiguriere Modell persistent (einmalig ~400ms, danach 0ms)."""
        self._configure_lock.acquire()  # SERIALIZE: Only one thread can configure at a time
        try:
            if name in self._active_ctx:
                logger.info(f"[CONFIGURE] {name}: bereits konfiguriert, skip")
                return'''

if old_start in code:
    code = code.replace(old_start, new_start)
    print('FIX 2: acquire() at start of _configure_model - OK')
    fixes += 1
else:
    print('FIX 2: ANCHOR NOT FOUND!')

# FIX 3: release() in finally of _configure_model (after existing finally)
# The method already has a try/finally for _configuring.set()
# We need to add another finally OUTSIDE that for the lock
old_end = '''        finally:
            self._configuring.set()

    def _unconfigure_model(self, name):'''

new_end = '''        finally:
            self._configuring.set()
        finally:
            self._configure_lock.release()

    def _unconfigure_model(self, name):'''

if old_end in code:
    code = code.replace(old_end, new_end)
    print('FIX 3: release() in finally of _configure_model - OK')
    fixes += 1
else:
    print('FIX 3: ANCHOR NOT FOUND!')

# FIX 4: acquire() at start of _unconfigure_model
old_unc_start = '''    def _unconfigure_model(self, name):
        """Entferne Modell."""
        if name not in self._active_ctx:
            return'''

new_unc_start = '''    def _unconfigure_model(self, name):
        """Entferne Modell."""
        self._configure_lock.acquire()  # SERIALIZE: Match _configure_model locking
        try:
            if name not in self._active_ctx:
                return'''

if old_unc_start in code:
    code = code.replace(old_unc_start, new_unc_start)
    print('FIX 4: acquire() at start of _unconfigure_model - OK')
    fixes += 1
else:
    print('FIX 4: ANCHOR NOT FOUND!')

# FIX 5: release() in finally of _unconfigure_model
# Find the end of _unconfigure_model (before next method)
old_unc_end = '''        logger.info(f"Modell freigegeben: {name}")

    def _ensure_configured(self, names):'''

new_unc_end = '''        logger.info(f"Modell freigegeben: {name}")
        finally:
            self._configure_lock.release()

    def _ensure_configured(self, names):'''

if old_unc_end in code:
    code = code.replace(old_unc_end, new_unc_end)
    print('FIX 5: release() in finally of _unconfigure_model - OK')
    fixes += 1
else:
    print('FIX 5: ANCHOR NOT FOUND!')

with open(svc, 'w') as f:
    f.write(code)

compile(open(svc).read(), svc, 'exec')
print(f'\n{fixes} Fixes. Syntax OK.')
print('\nJETZT: sudo systemctl restart moloch.service')
