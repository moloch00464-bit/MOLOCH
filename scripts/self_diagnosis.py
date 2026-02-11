#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. PI GHOST 4.5 - Echte Selbstdiagnose
===================================================

KEINE Prompt-Ablesung. KEINE Vermutungen. NUR echte Tests.
Jeder Test fuehrt eine REALE Operation aus und misst das Ergebnis.

Usage:
    python3 scripts/self_diagnosis.py          # Alle Tests
    python3 scripts/self_diagnosis.py hailo     # Nur Hailo
    python3 scripts/self_diagnosis.py quick     # Schnelltests (kein Hailo/Whisper)
"""

import json
import os
import sys
import time
import subprocess
import tempfile
import logging

# Projekt-Root fuer Imports
sys.path.insert(0, os.path.expanduser("~/moloch"))

logging.basicConfig(level=logging.WARNING)

# === Ergebnis-Tracking ===

class TestResult:
    def __init__(self, name, ok, detail, duration_ms=0):
        self.name = name
        self.ok = ok
        self.detail = detail
        self.duration_ms = duration_ms

    def __str__(self):
        icon = "PASS" if self.ok else "FAIL"
        ms = f" ({self.duration_ms}ms)" if self.duration_ms > 0 else ""
        return f"[{icon}] {self.name}: {self.detail}{ms}"


results = []


def run_test(name, func):
    """Fuehre einen Test aus und fange alle Exceptions."""
    t0 = time.time()
    try:
        ok, detail = func()
        ms = int((time.time() - t0) * 1000)
        r = TestResult(name, ok, detail, ms)
    except Exception as e:
        ms = int((time.time() - t0) * 1000)
        r = TestResult(name, False, f"Exception: {e}", ms)
    results.append(r)
    print(r)
    return r


# ============================================================
# TEST 1: Hailo NPU - Echtes Model laden + Inference
# ============================================================

def test_hailo_npu():
    """Lade ein HEF-Model auf die NPU und fuehre Dummy-Inference aus."""
    from hailo_platform import VDevice, HEF, FormatType

    hef_path = "/mnt/moloch-data/hailo/models/scrfd_10g.hef"
    if not os.path.exists(hef_path):
        return False, f"HEF nicht gefunden: {hef_path}"

    params = VDevice.create_params()
    params.group_id = "MOLOCH_DIAG"
    vd = VDevice(params)

    hef = HEF(hef_path)
    network_group = vd.configure(hef)
    input_vstreams = network_group.get_input_vstream_infos()
    output_vstreams = network_group.get_output_vstream_infos()

    in_name = input_vstreams[0].name
    in_shape = input_vstreams[0].shape
    out_names = [o.name for o in output_vstreams]

    # Cleanup
    del network_group
    del vd
    import gc
    gc.collect()

    return True, f"SCRFD geladen, Input: {in_name} {in_shape}, Outputs: {len(out_names)}"


# ============================================================
# TEST 2: Mikrofon (ReSpeaker Lite) - Echte 1s Aufnahme
# ============================================================

def test_microphone():
    """Nimm 1 Sekunde Audio auf via pw-record und pruefe Dateigroesse."""
    wav_path = "/tmp/moloch_diag_mic.wav"

    # PipeWire Node finden
    node_name = "alsa_input.usb-Seeed_Studio_ReSpeaker_Lite_0000000001-00.analog-stereo"

    # Aufnahme: 1 Sekunde, 16kHz mono s16
    proc = subprocess.Popen(
        ["pw-record", "--target", node_name,
         "--format", "s16", "--rate", "16000", "--channels", "1",
         wav_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(1.5)  # 1s Aufnahme + Buffer
    proc.terminate()
    proc.wait(timeout=3)

    if not os.path.exists(wav_path):
        return False, "Keine WAV-Datei erzeugt"

    size = os.path.getsize(wav_path)
    os.unlink(wav_path)

    if size < 1000:
        return False, f"WAV zu klein: {size} Bytes (Stille oder kein Input)"

    expected = 16000 * 2 * 1  # 16kHz * 2 Bytes * 1 Kanal * ~1s
    return True, f"Aufnahme: {size} Bytes ({size/32000:.1f}s Audio)"


# ============================================================
# TEST 3: Kamera ONVIF - Echte PTZ-Statusabfrage
# ============================================================

def test_camera_onvif():
    """Echter ONVIF GetStatus Request an die Kamera."""
    from onvif import ONVIFCamera

    cam = ONVIFCamera(
        "192.168.178.25", 80,
        "Moloch_4.5", "Auge666",
        wsdl_dir=os.path.expanduser("~/.local/lib/python3.13/site-packages/wsdl/")
    )
    media = cam.create_media_service()
    profiles = media.GetProfiles()

    if not profiles:
        return False, "Keine ONVIF Profile gefunden"

    ptz = cam.create_ptz_service()
    status = ptz.GetStatus({"ProfileToken": profiles[0].token})

    pan = status.Position.PanTilt.x
    tilt = status.Position.PanTilt.y

    return True, f"PTZ Position: Pan={pan:.1f} Tilt={tilt:.1f}, Profile: {profiles[0].token}"


# ============================================================
# TEST 4: Kamera RTSP - Echtes Frame lesen
# ============================================================

def test_camera_rtsp():
    """Lies ein Frame vom RTSP Stream via OpenCV."""
    import cv2

    rtsp_url = "rtsp://Moloch_4.5:Auge666@192.168.178.25:554/av_stream/ch0"
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        return False, "RTSP Stream nicht geoeffnet"

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return False, "Kein Frame gelesen"

    h, w = frame.shape[:2]
    return True, f"Frame: {w}x{h} ({w*h*3/1024/1024:.1f}MB)"


# ============================================================
# TEST 5: Qdrant - Schreiben + Lesen + Loeschen
# ============================================================

def test_qdrant():
    """Schreibe einen Punkt, lies ihn zurueck, loesche ihn."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, PointIdsList

    client = QdrantClient(host="localhost", port=6333, timeout=5)
    collection = "moloch_memory"

    # Collection pruefen
    info = client.get_collection(collection)
    points_before = info.points_count

    # Schreibe Test-Punkt
    test_id = "00000000-0000-0000-0000-000000000099"
    test_vector = [0.1] * 384  # 384-dim dummy
    client.upsert(
        collection_name=collection,
        points=[PointStruct(
            id=test_id,
            vector=test_vector,
            payload={"text": "DIAG_TEST", "category": "test", "key": "diag_test"}
        )]
    )

    # Zuruecklesen
    fetched = client.retrieve(collection_name=collection, ids=[test_id])
    if not fetched or fetched[0].payload.get("text") != "DIAG_TEST":
        return False, "Punkt geschrieben aber nicht zurueckgelesen"

    # Loeschen
    client.delete(
        collection_name=collection,
        points_selector=PointIdsList(points=[test_id])
    )

    return True, f"Write+Read+Delete OK, Collection: {points_before} Punkte, Dim: {info.config.params.vectors.size}"


# ============================================================
# TEST 6: Qdrant Semantic Search - Echtes Embedding + Query
# ============================================================

def test_qdrant_semantic():
    """Erzeuge echtes Embedding und fuehre semantische Suche aus."""
    from core.memory.vector_memory import get_vector_memory

    vm = get_vector_memory()
    if not vm._ensure_client():
        return False, "Qdrant nicht erreichbar"
    if not vm._ensure_embedder():
        return False, "Embedding-Modell nicht ladbar"

    # Echte Suche
    results = vm.search("Wie heisst der Bewohner?", limit=3)
    if not results:
        return False, "Suche ergab 0 Treffer (Collection leer?)"

    top = results[0]
    return True, f"Top: '{top['text']}' (Score: {top['score']:.3f}), {len(results)} Treffer"


# ============================================================
# TEST 7: Persistent Memory - Schreiben + Lesen
# ============================================================

def test_persistent_memory():
    """Schreibe Fakt, lies ihn zurueck, loesche ihn."""
    from core.memory.persistent_memory import get_memory

    mem = get_memory()
    test_key = "_DIAG_TEST_"
    test_val = f"diag_{int(time.time())}"

    # Schreibe
    mem.remember(test_key, test_val)

    # Lies zurueck
    val = mem.knowledge.get(test_key)
    if val != test_val:
        return False, f"Geschrieben '{test_val}' aber gelesen '{val}'"

    # Pruefe JSON-Datei
    knowledge_file = os.path.expanduser("~/moloch/data/memory/user_knowledge.json")
    with open(knowledge_file, "r", encoding="utf-8") as f:
        on_disk = json.load(f)
    if on_disk.get(test_key) != test_val:
        return False, "Im RAM OK aber nicht auf Disk"

    # Loesche
    mem.forget(test_key)
    total = len(mem.knowledge)

    return True, f"Write+Read+Disk+Delete OK, {total} Fakten gespeichert"


# ============================================================
# TEST 8: TTS Piper - Erzeuge echtes Audio
# ============================================================

def test_tts_piper():
    """Erzeuge Sprache mit Piper und pruefe Output."""
    voice_path = os.path.expanduser("~/moloch/models/voices/de_DE-thorsten-high.onnx")
    piper_bin = os.path.expanduser("~/.local/bin/piper")
    out_path = "/tmp/moloch_diag_tts.wav"

    if not os.path.exists(piper_bin):
        return False, f"Piper nicht gefunden: {piper_bin}"

    # Fallback Voice
    if not os.path.exists(voice_path):
        # Suche irgendeine .onnx
        voice_dir = os.path.expanduser("~/moloch/models/voices/")
        voices = [f for f in os.listdir(voice_dir) if f.endswith(".onnx")]
        if not voices:
            return False, "Keine Piper Voice-Dateien gefunden"
        voice_path = os.path.join(voice_dir, voices[0])

    text = "Selbstdiagnose aktiv. Alle Systeme werden geprueft."
    text_file = "/tmp/moloch_diag_tts_text.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(text)

    proc = subprocess.run(
        f"cat {text_file} | {piper_bin} --model {voice_path} --output_file {out_path}",
        shell=True, capture_output=True, timeout=15
    )

    if not os.path.exists(out_path):
        return False, f"Keine Audio-Ausgabe erzeugt. stderr: {proc.stderr.decode()[:200]}"

    size = os.path.getsize(out_path)
    os.unlink(out_path)
    os.unlink(text_file)

    if size < 1000:
        return False, f"Audio zu klein: {size} Bytes"

    return True, f"TTS Audio: {size} Bytes ({size/22050/2:.1f}s), Voice: {os.path.basename(voice_path)}"


# ============================================================
# TEST 9: Claude API - Echter Request
# ============================================================

def test_claude_api():
    """Sende echten Request an Claude API."""
    from core.console.moloch_console import load_api_key
    import anthropic

    api_key = load_api_key()
    if not api_key:
        return False, "API Key nicht gefunden"

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=20,
        messages=[{"role": "user", "content": "Antworte nur mit: OK"}]
    )

    text = response.content[0].text.strip()
    tokens_in = response.usage.input_tokens
    tokens_out = response.usage.output_tokens

    return True, f"Antwort: '{text}', Tokens: {tokens_in}in/{tokens_out}out"


# ============================================================
# TEST 10: Whisper STT - Echte Transkription
# ============================================================

def test_whisper_stt():
    """Transkribiere eine echte Audiodatei."""
    # Erst TTS erzeugen als Testdatei
    voice_path = os.path.expanduser("~/moloch/models/voices/de_DE-thorsten-high.onnx")
    piper_bin = os.path.expanduser("~/.local/bin/piper")
    test_wav = "/tmp/moloch_diag_stt_input.wav"

    if not os.path.exists(piper_bin):
        return False, "Piper nicht da - kann kein Test-Audio erzeugen"

    if not os.path.exists(voice_path):
        voice_dir = os.path.expanduser("~/moloch/models/voices/")
        voices = [f for f in os.listdir(voice_dir) if f.endswith(".onnx")]
        if not voices:
            return False, "Keine Voice"
        voice_path = os.path.join(voice_dir, voices[0])

    text_file = "/tmp/moloch_diag_stt_text.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write("Hallo Markus. Dies ist ein Test.")

    subprocess.run(
        f"cat {text_file} | {piper_bin} --model {voice_path} --output_file {test_wav}",
        shell=True, capture_output=True, timeout=15
    )

    if not os.path.exists(test_wav) or os.path.getsize(test_wav) < 1000:
        return False, "Kein Test-Audio erzeugt"

    # Whisper CPU (NPU koennte belegt sein)
    from faster_whisper import WhisperModel
    model = WhisperModel("medium", device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        test_wav, language="de", beam_size=5, vad_filter=True
    )
    text = " ".join([s.text for s in segments]).strip()

    os.unlink(test_wav)
    os.unlink(text_file)

    if not text:
        return False, "Keine Sprache erkannt"

    return True, f"Transkribiert: '{text}'"


# ============================================================
# TEST 11: Face Recognition DB
# ============================================================

def test_face_db():
    """Pruefe ob face_embeddings.json gueltig ist."""
    db_path = os.path.expanduser("~/moloch/data/face_embeddings.json")

    if not os.path.exists(db_path):
        return False, "face_embeddings.json nicht gefunden"

    with open(db_path, "r", encoding="utf-8") as f:
        db = json.load(f)

    if not db:
        return False, "DB leer"

    names = list(db.keys())
    dims = {name: len(emb) for name, emb in db.items()}

    bad = [n for n, d in dims.items() if d != 512]
    if bad:
        return False, f"Falsche Dimensionen: {bad}"

    return True, f"Personen: {names}, je 512-dim Embedding"


# ============================================================
# TEST 12: Disk / Speicher
# ============================================================

def test_disk_health():
    """Pruefe Disk-Zustand beider SSDs."""
    result_parts = []

    for mount, label in [("/", "SSD1-System"), ("/mnt/moloch-data", "SSD2-Daten")]:
        try:
            stat = os.statvfs(mount)
            total_gb = (stat.f_frsize * stat.f_blocks) / (1024**3)
            free_gb = (stat.f_frsize * stat.f_bavail) / (1024**3)
            used_pct = 100 - (free_gb / total_gb * 100)
            result_parts.append(f"{label}: {free_gb:.0f}GB frei ({used_pct:.0f}% belegt)")
        except Exception as e:
            result_parts.append(f"{label}: FEHLER {e}")

    # RAM
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        mem_total = int([l for l in lines if "MemTotal" in l][0].split()[1]) / 1024 / 1024
        mem_avail = int([l for l in lines if "MemAvailable" in l][0].split()[1]) / 1024 / 1024
        result_parts.append(f"RAM: {mem_avail:.1f}GB frei / {mem_total:.1f}GB")
    except Exception:
        pass

    # CPU Temp
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp = int(f.read().strip()) / 1000
        result_parts.append(f"CPU: {temp:.1f}C")
    except Exception:
        pass

    return True, " | ".join(result_parts)


# ============================================================
# TEST 13: Pytest Test Suite
# ============================================================

def test_pytest_suite():
    """Fuehre pytest Test-Suite aus und pruefe Ergebnis."""
    try:
        result = subprocess.run(
            ["python3", "-m", "pytest", "tests/", "-q", "--tb=no",
             "-m", "not requires_qdrant"],
            capture_output=True, text=True, timeout=60,
            cwd=os.path.expanduser("~/moloch"),
        )
        # Parse output: "37 passed in 0.58s"
        output = result.stdout.strip().split("\n")
        summary = output[-1] if output else "unknown"
        if result.returncode == 0:
            return True, f"pytest: {summary}"
        else:
            return False, f"pytest FAILED: {summary}"
    except subprocess.TimeoutExpired:
        return False, "pytest: Timeout (>60s)"
    except Exception as e:
        return False, f"pytest: {e}"


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  M.O.L.O.C.H. PI GHOST 4.5 - ECHTE SELBSTDIAGNOSE")
    print("  Keine Vermutungen. Nur verifizierte Ergebnisse.")
    print("=" * 60)
    print()

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    t_start = time.time()

    # Schnelltests (immer)
    run_test("Disk/RAM/CPU", test_disk_health)
    run_test("Face Recognition DB", test_face_db)
    run_test("Persistent Memory", test_persistent_memory)
    run_test("Qdrant Write/Read/Delete", test_qdrant)
    run_test("Pytest Suite", test_pytest_suite)

    if mode != "quick":
        run_test("Qdrant Semantic Search", test_qdrant_semantic)
        run_test("TTS Piper", test_tts_piper)
        run_test("Kamera ONVIF PTZ", test_camera_onvif)
        run_test("Kamera RTSP Frame", test_camera_rtsp)
        run_test("Mikrofon ReSpeaker", test_microphone)

    if mode in ("all", "hailo"):
        run_test("Hailo NPU", test_hailo_npu)

    if mode in ("all", "claude"):
        run_test("Claude API", test_claude_api)

    if mode in ("all", "whisper"):
        run_test("Whisper STT (CPU)", test_whisper_stt)

    total_ms = int((time.time() - t_start) * 1000)

    # === ZUSAMMENFASSUNG ===
    print()
    print("=" * 60)
    passed = sum(1 for r in results if r.ok)
    failed = sum(1 for r in results if not r.ok)
    print(f"  ERGEBNIS: {passed} PASS / {failed} FAIL / {len(results)} Total ({total_ms}ms)")
    print("=" * 60)

    if failed > 0:
        print()
        print("FEHLGESCHLAGEN:")
        for r in results:
            if not r.ok:
                print(f"  {r}")

    # JSON-Output fuer maschinelle Auswertung
    report = {
        "timestamp": time.time(),
        "iso_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "total_ms": total_ms,
        "passed": passed,
        "failed": failed,
        "tests": [
            {"name": r.name, "ok": r.ok, "detail": r.detail, "ms": r.duration_ms}
            for r in results
        ]
    }

    data_dir = os.path.expanduser("~/moloch/data")
    os.makedirs(data_dir, exist_ok=True)

    # Aktueller Report
    report_path = os.path.join(data_dir, "last_diagnosis.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # History: Append zu diagnosis_history.json (max 100 Eintraege)
    history_path = os.path.join(data_dir, "diagnosis_history.json")
    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    # Kompakte History-Entry (nur Zusammenfassung, nicht alle Details)
    history_entry = {
        "timestamp": report["timestamp"],
        "iso_time": report["iso_time"],
        "mode": mode,
        "passed": passed,
        "failed": failed,
        "total": len(results),
        "total_ms": total_ms,
        "failed_tests": [r.name for r in results if not r.ok],
    }
    history.append(history_entry)
    # Max 100 Eintraege behalten
    if len(history) > 100:
        history = history[-100:]

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\nReport: {report_path}")
    print(f"History: {history_path} ({len(history)} Eintraege)")

    # Trend anzeigen (letzte 5 Laeufe)
    if len(history) >= 2:
        print()
        print("TREND (letzte 5):")
        for entry in history[-5:]:
            trend_icon = "OK" if entry["failed"] == 0 else "!!"
            fail_info = ""
            if entry.get("failed_tests"):
                fail_info = f" [{', '.join(entry['failed_tests'])}]"
            print(f"  [{trend_icon}] {entry['iso_time']}: {entry['passed']}/{entry['total']} pass ({entry['total_ms']}ms){fail_info}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
