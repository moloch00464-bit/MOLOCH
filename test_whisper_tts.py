#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, time, subprocess
sys.path.insert(0, '/home/molochzuhause/moloch')

PIPER = '/home/molochzuhause/.local/bin/piper'
MODEL = '/home/molochzuhause/.local/share/piper-voices/de_DE-thorsten-high/de_DE-thorsten-high.onnx'

def record_audio(duration=4):
    f = '/tmp/whisper_test.wav'
    print('Aufnahme startet in 3 Sekunden...')
    time.sleep(3)
    print('AUFNAHME LAEUFT...')
    p = subprocess.Popen(['pw-record','--rate','16000','--channels','1','--format','s16',f])
    time.sleep(duration)
    p.terminate()
    p.wait()
    print('Aufnahme beendet')
    sz = os.path.getsize(f)
    print(f'Datei: {sz} bytes')
    return f if sz > 1000 else None

def transcribe(f):
    print('Transkribiere...')
    from core.speech.hailo_whisper import get_whisper
    w = get_whisper()
    print(f'Backend: {w.backend}')
    t = w.transcribe(f, language='de')
    print(f'Erkannt: {t}' if t else 'Nichts erkannt')
    return t

def speak(text):
    print(f'TTS: {text}')
    wav = '/tmp/tts_out.wav'
    with open('/tmp/tts_in.txt','w') as f:
        f.write(text)
    subprocess.run(f'cat /tmp/tts_in.txt | {PIPER} --model {MODEL} --output_file {wav} 2>/dev/null && aplay {wav} 2>/dev/null', shell=True)

def main():
    print('='*50)
    print('  WHISPER + TTS TEST')
    print('='*50)
    af = record_audio(4)
    if not af:
        print('Aufnahme fehlgeschlagen!')
        return
    t = transcribe(af)
    if not t or t.strip() in ['.','!','']:
        t = 'Ich konnte dich nicht verstehen'
    speak(f'Du hast gesagt: {t}')
    print('FERTIG')

if __name__ == '__main__':
    main()
