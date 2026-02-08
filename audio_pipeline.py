#\!/usr/bin/env python3
"""
M.O.L.O.C.H. Audio Pipeline
SmartMic → Whisper STT → Processing → TTS → HDMI

Usage: python3 audio_pipeline.py [--model tiny|base|small|medium]
"""

import subprocess
import tempfile
import sys
import os
from pathlib import Path

# Konfiguration
WHISPER_MODEL = "small"  # Kompromiss zwischen Speed und Qualität
RECORD_SECONDS = 5
SAMPLE_RATE = 16000
BT_SOURCE_ID = "58"  # SmartMic bluez_input

def record_audio(duration: int = RECORD_SECONDS) -> str:
    """Nimmt Audio vom SmartMic auf, gibt Pfad zur WAV-Datei zurück."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    
    cmd = [
        "pw-record",
        "--target", BT_SOURCE_ID,
        "--channels", "1",
        "--rate", str(SAMPLE_RATE),
        wav_path
    ]
    
    print(f"🎤 Aufnahme läuft ({duration}s)... Sprich jetzt\!")
    try:
        subprocess.run(cmd, timeout=duration + 1, check=False)
    except subprocess.TimeoutExpired:
        # Normal - wir stoppen nach timeout
        subprocess.run(["pkill", "-f", f"pw-record.*{wav_path}"], check=False)
    
    return wav_path

def transcribe(wav_path: str, model: str = WHISPER_MODEL) -> str:
    """Transkribiert Audio mit faster-whisper."""
    print(f"🧠 Transkribiere mit Whisper ({model})...")
    
    from faster_whisper import WhisperModel
    
    # Model laden (cached)
    whisper = WhisperModel(model, device="cpu", compute_type="int8")
    
    segments, info = whisper.transcribe(wav_path, language="de")
    text = " ".join([seg.text for seg in segments]).strip()
    
    return text

def speak(text: str):
    """Gibt Text über TTS auf HDMI aus."""
    if not text:
        text = "Ich habe nichts verstanden."
    
    print(f"🔊 Sage: {text}")
    subprocess.run([
        "/home/molochzuhause/bin/moloch-speak",
        text
    ], check=False)

def main():
    model = WHISPER_MODEL
    if len(sys.argv) > 2 and sys.argv[1] == "--model":
        model = sys.argv[2]
    
    print("=" * 50)
    print("M.O.L.O.C.H. AUDIO PIPELINE")
    print("=" * 50)
    
    # Begrüßung
    speak("M.O.L.O.C.H. Spracherkennung aktiv.")
    
    while True:
        try:
            # 1. Audio aufnehmen
            wav_path = record_audio()
            
            # 2. Transkribieren
            text = transcribe(wav_path, model)
            print(f"📝 Erkannt: {text}")
            
            # 3. Cleanup
            os.unlink(wav_path)
            
            # 4. Verarbeiten & Antworten
            if text.lower() in ["stop", "stopp", "beenden", "exit"]:
                speak("M.O.L.O.C.H. Spracherkennung beendet.")
                break
            
            # Echo-Modus (später: LLM Integration)
            if text:
                speak(f"Du hast gesagt: {text}")
            else:
                speak("Keine Sprache erkannt.")
                
        except KeyboardInterrupt:
            print("\n⏹️ Abgebrochen")
            speak("Auf Wiedersehen.")
            break
        except Exception as e:
            print(f"❌ Fehler: {e}")
            continue

if __name__ == "__main__":
    main()
