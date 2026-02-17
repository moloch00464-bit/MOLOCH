#!/usr/bin/env python3
"""Fix: Model Configuration Race Condition - Version 2

PROBLEM:
- Idle Pre-Load and Perception Engine try to configure models simultaneously
- No serialization -> 3-4 models active -> VIOLATION -> timeout -> hung

SIMPLER FIX:
- Add _configure_semaphore (Semaphore(1) = mutex)
- acquire() before checking if already configured
- release() at ALL exit points (early return + normal end + except)

This prevents racing without complex indentation changes.
"""

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

fixes = 0

# FIX 1: Add _configure_semaphore in __init__
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
        self._configure_sem = threading.Semaphore(1)  # Serialize configure/unconfigure (race protection)'''

if old_init in code:
    code = code.replace(old_init, new_init)
    print('FIX 1: Add _configure_semaphore - OK')
    fixes += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

# FIX 2: acquire() at very start of _configure_model
old_start = '''    def _configure_model(self, name):
        """Konfiguriere Modell persistent (einmalig ~400ms, danach 0ms)."""
        if name in self._active_ctx:'''

new_start = '''    def _configure_model(self, name):
        """Konfiguriere Modell persistent (einmalig ~400ms, danach 0ms)."""
        self._configure_sem.acquire()  # SERIALIZE: Only one thread can configure at a time
        if name in self._active_ctx:
            self._configure_sem.release()'''

if old_start in code:
    code = code.replace(old_start, new_start)
    print('FIX 2: acquire() at start + release() on early return 1 - OK')
    fixes += 1
else:
    print('FIX 2: ANCHOR NOT FOUND!')

# FIX 3: release() on second early return
old_check = '''        if name not in self._models:
            logger.warning(f"[CONFIGURE] {name}: nicht in self._models")
            return'''

new_check = '''        if name not in self._models:
            logger.warning(f"[CONFIGURE] {name}: nicht in self._models")
            self._configure_sem.release()
            return'''

if old_check in code:
    code = code.replace(old_check, new_check)
    print('FIX 3: release() on early return 2 - OK')
    fixes += 1
else:
    print('FIX 3: ANCHOR NOT FOUND!')

# FIX 4: release() in finally block (after _configuring.set())
old_finally = '''        finally:
            self._configuring.set()

    def _unconfigure_model(self, name):'''

new_finally = '''        finally:
            self._configuring.set()
            self._configure_sem.release()  # Release configure lock

    def _unconfigure_model(self, name):'''

if old_finally in code:
    code = code.replace(old_finally, new_finally)
    print('FIX 4: release() in finally block - OK')
    fixes += 1
else:
    print('FIX 4: ANCHOR NOT FOUND!')

# FIX 5: Same pattern for _unconfigure_model
old_unc = '''    def _unconfigure_model(self, name):
        """Entferne Modell."""
        if name not in self._active_ctx:
            return'''

new_unc = '''    def _unconfigure_model(self, name):
        """Entferne Modell."""
        self._configure_sem.acquire()  # SERIALIZE
        if name not in self._active_ctx:
            self._configure_sem.release()
            return'''

if old_unc in code:
    code = code.replace(old_unc, new_unc)
    print('FIX 5: Lock _unconfigure_model + early return - OK')
    fixes += 1
else:
    print('FIX 5: ANCHOR NOT FOUND!')

# FIX 6: release() at end of _unconfigure_model
old_unc_end = '''        logger.info(f"Modell freigegeben: {name}")

    def _ensure_configured(self, names):'''

new_unc_end = '''        logger.info(f"Modell freigegeben: {name}")
        self._configure_sem.release()  # Release configure lock

    def _ensure_configured(self, names):'''

if old_unc_end in code:
    code = code.replace(old_unc_end, new_unc_end)
    print('FIX 6: release() at end of _unconfigure_model - OK')
    fixes += 1
else:
    print('FIX 6: ANCHOR NOT FOUND!')

with open(svc, 'w') as f:
    f.write(code)

try:
    compile(open(svc).read(), svc, 'exec')
    print(f'\n{fixes}/6 Fixes. Syntax OK.')
    print('\nJETZT: sudo systemctl restart moloch.service')
except SyntaxError as e:
    print(f'\nSYNTAX ERROR: {e}')
    print('Rolling back...')
    import subprocess
    subprocess.run(['git', 'checkout', svc], cwd='/home/molochzuhause/moloch')
