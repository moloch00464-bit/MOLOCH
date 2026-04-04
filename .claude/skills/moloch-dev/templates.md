# Code-Templates

## HailoRT On-Demand Processor

```python
class MyProcessor:
    def __init__(self):
        self._lock = threading.Lock()
        self._vdevice = None
        self._configured = None
        self._loaded = False
        self._load_error = None

    def _ensure_loaded(self) -> bool:
        if self._loaded: return True
        if self._load_error: return False
        try:
            import hailo_platform as hp
            from hailo_platform.pyhailort._pyhailort import FormatType
            params = hp.VDevice.create_params()
            params.group_id = "SHARED"           # PFLICHT!
            self._vdevice = hp.VDevice(params)
            model = self._vdevice.create_infer_model(HEF_PATH)
            for n in model.output_names:
                model.output(n).set_format_type(FormatType.FLOAT32)
            self._configured = model.configure()
            self._loaded = True
            return True
        except Exception as e:
            self._load_error = str(e)
            return False

    def process(self, img_rgb):
        with self._lock:
            if not self._ensure_loaded(): return img_rgb
            try:
                inp = preprocess(img_rgb)        # uint8!
                bindings = self._configured.create_bindings()
                bindings.input().set_buffer(np.ascontiguousarray(inp))
                out_buf = np.empty(out_shape, dtype=np.float32)
                bindings.output(name).set_buffer(out_buf)
                self._configured.run([bindings], TIMEOUT_MS)
                return postprocess(out_buf)
            except Exception:
                return img_rgb  # Fallback: Original
```

## GStreamer RGB/BGR Konvertierung

```python
# GStreamer = RGB, cv2 = BGR
frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(path, frame_bgr)
```

## Singleton Pattern

```python
_instance = None
_lock = threading.Lock()

def get_thing() -> "Thing":
    global _instance
    with _lock:
        if _instance is None:
            _instance = Thing()
    return _instance
```

## Safe JSON Write (Atomic + NTFS-Fallback)

```python
def safe_json_write(path: str, data) -> None:
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        try: os.unlink(tmp)
        except OSError: pass
```

## Subprocess mit Timeout

```python
def safe_run(cmd: list, timeout: int = 30):
    return subprocess.run(cmd, capture_output=True, timeout=timeout, text=True)
```
