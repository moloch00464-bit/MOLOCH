#!/usr/bin/env python3
"""
Whisper Test - SmartMic + NPU
Mit Anti-Halluzination
"""

import subprocess
import sys
import os
import tempfile
import signal
import threading
import re

# Fix UTF-8 encoding für Umlaute
os.environ['PYTHONIOENCODING'] = 'utf-8'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RATE = 16000
MAX_DURATION = 30  # Max 30 Sekunden
recording_process = None
stop_flag = False

def record_until_enter(output_path: str):
    global recording_process, stop_flag
    recording_process = subprocess.Popen(
        ['pw-record', '--rate', str(RATE), '--channels', '1', '--format', 's16', output_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    import time
    start = time.time()
    while not stop_flag and recording_process.poll() is None:
        if time.time() - start > MAX_DURATION:
            print(f'(Max {MAX_DURATION}s erreicht)')
            break
        time.sleep(0.1)
    if recording_process.poll() is None:
        recording_process.send_signal(signal.SIGINT)
        recording_process.wait(timeout=2)

def process_audio(input_path: str, output_path: str):
    """Stille entfernen, auf 5s begrenzen, normalisieren."""
    subprocess.run([
        'sox', input_path, output_path,
        'silence', '1', '0.1', '2%',
        'reverse',
        'silence', '1', '0.1', '2%',
        'reverse',
        'trim', '0', str(MAX_DURATION),
        'norm', '-3'
    ], capture_output=True)

def remove_repetitions(text: str) -> str:
    """Entferne wiederholte Woerter/Phrasen."""
    if not text:
        return text
    words = text.split()
    if len(words) < 3:
        return text
    # Wenn ein Wort mehr als 3x vorkommt, nur 1-2x behalten
    from collections import Counter
    counts = Counter(words)
    result = []
    seen_counts = Counter()
    for word in words:
        seen_counts[word] += 1
        if counts[word] <= 3 or seen_counts[word] <= 2:
            result.append(word)
    return ' '.join(result)

def transcribe_npu(wav_path: str) -> str:
    try:
        from core.speech.hailo_whisper import get_whisper
        whisper = get_whisper()
        return whisper.transcribe(wav_path, language='de') or ''
    except Exception as e:
        return f'[Fehler: {e}]'

def main():
    global stop_flag
    
    print()
    print('=' * 50)
    print('  WHISPER TEST - SmartMic + NPU')
    print('  Max 30 Sekunden, Anti-Halluzination')
    print('=' * 50)
    
    print('Lade NPU-Whisper...')
    from core.speech.hailo_whisper import get_whisper
    npu_whisper = get_whisper()
    print(f'Backend: {npu_whisper.backend}')
    print()
    
    try:
        while True:
            input('ENTER zum Aufnehmen: ')
            
            with tempfile.TemporaryDirectory() as tmpdir:
                raw_path = os.path.join(tmpdir, 'raw.wav')
                clean_path = os.path.join(tmpdir, 'clean.wav')
                
                stop_flag = False
                rec_thread = threading.Thread(target=record_until_enter, args=(raw_path,))
                rec_thread.start()
                
                print(f'AUFNAHME... (ENTER oder max {MAX_DURATION}s)')
                input()
                
                stop_flag = True
                rec_thread.join(timeout=3)
                
                if not os.path.exists(raw_path):
                    print('Keine Aufnahme!')
                    continue
                    
                raw_size = os.path.getsize(raw_path)
                print(f'Roh: {raw_size/1024:.1f} KB')
                
                if raw_size < 1000:
                    print('Zu kurz!')
                    continue
                
                process_audio(raw_path, clean_path)
                
                if not os.path.exists(clean_path) or os.path.getsize(clean_path) < 500:
                    print('Nur Stille!')
                    continue
                
                print('Transkribiere...')
                raw_result = npu_whisper.transcribe(clean_path, language='de') or ''
                raw_result = raw_result.encode("utf-8", "replace").decode("utf-8")
                result = remove_repetitions(raw_result)
                
                print()
                print('-' * 50)
                if result and result.strip() not in ['.', '!', '']:
                    print(f'  "{result}"')
                    if raw_result != result:
                        print(f'  (bereinigt von: "{raw_result[:50]}...")')
                else:
                    print('  (nichts erkannt)')
                print('-' * 50)
                print()
                
    except KeyboardInterrupt:
        print('Beendet.')
        return 0

if __name__ == '__main__':
    sys.exit(main())
