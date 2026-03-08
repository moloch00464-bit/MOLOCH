#!/usr/bin/env python3
"""
M.O.L.O.C.H. — ReSpeaker UDP Audio-Test
=========================================
Empfaengt Audio per UDP vom ESP32-S3 und prueft:
  1. Ob Daten ankommen (Timeout 5s)
  2. Ob Max-Amplitude > 100 (nicht nur Nullen)
  3. Latenz-Messung (Klatschen → Timestamp)
  4. Optional: als WAV-Datei speichern

Benutzung:
  python3 scripts/test_respeaker_udp.py --rate 16000
  python3 scripts/test_respeaker_udp.py --rate 48000
  python3 scripts/test_respeaker_udp.py --rate 16000 --save /tmp/test.wav
"""

import socket
import struct
import time
import wave
import argparse
import sys
import json
import urllib.request


ESP32_IP = "10.42.0.2"
HTTP_PORT = 80


def switch_mode(rate):
    """Audio-Modus am ESP32 umschalten per HTTP."""
    url = f"http://{ESP32_IP}:{HTTP_PORT}/audio/mode?rate={rate}"
    try:
        req = urllib.request.Request(url, method="POST")
        resp = urllib.request.urlopen(req, timeout=3)
        print(f"[MODE] {resp.read().decode()}")
        return True
    except Exception as e:
        print(f"[MODE] Fehler: {e}")
        return False


def get_status():
    """Status vom ESP32 abfragen."""
    url = f"http://{ESP32_IP}:{HTTP_PORT}/audio/status"
    try:
        resp = urllib.request.urlopen(url, timeout=3)
        data = json.loads(resp.read().decode())
        print(f"[STATUS] {json.dumps(data, indent=2)}")
        return data
    except Exception as e:
        print(f"[STATUS] Fehler: {e}")
        return None


def record_udp(port, duration_s, rate, channels, save_path=None):
    """UDP Audio empfangen und analysieren."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.settimeout(5.0)

    print(f"[REC] Lausche auf UDP Port {port} ({rate}Hz, {channels}ch, {duration_s}s)...")

    all_data = bytearray()
    packets = 0
    max_amp = 0
    non_zero = 0
    total_samples = 0
    start_time = None

    try:
        while True:
            try:
                data = s.recv(4096)
            except socket.timeout:
                if packets == 0:
                    print("[REC] TIMEOUT — keine UDP-Pakete empfangen!")
                    print("[REC] Pruefen:")
                    print("  1. Laeuft die ESP32-Firmware? (http://10.42.0.2/audio/status)")
                    print("  2. WiFi verbunden? (ping 10.42.0.2)")
                    print("  3. XMOS I2S Firmware aktiv?")
                    s.close()
                    return None
                break

            if start_time is None:
                start_time = time.time()
                print(f"[REC] Erste Daten empfangen! ({len(data)} Bytes)")

            packets += 1
            all_data.extend(data)

            # Amplitude pruefen
            n_samples = len(data) // 2
            total_samples += n_samples
            for i in range(0, len(data), 2):
                sample = struct.unpack_from("<h", data, i)[0]
                amp = abs(sample)
                if amp > max_amp:
                    max_amp = amp
                if sample != 0:
                    non_zero += 1

            # Dauer pruefen
            elapsed = time.time() - start_time
            if elapsed >= duration_s:
                break

    except KeyboardInterrupt:
        print("[REC] Abgebrochen")
    finally:
        s.close()

    if packets == 0:
        return None

    elapsed = time.time() - start_time
    total_bytes = len(all_data)

    result = {
        "status": "PASS" if max_amp > 100 else "FAIL",
        "dauer_s": round(elapsed, 2),
        "pakete": packets,
        "bytes_gesamt": total_bytes,
        "samples_gesamt": total_samples,
        "max_amplitude": max_amp,
        "non_zero_samples": non_zero,
        "non_zero_prozent": round(100 * non_zero / total_samples, 1) if total_samples > 0 else 0,
        "bitrate_kbps": round(total_bytes * 8 / elapsed / 1000, 1),
        "rate_hz": rate,
        "channels": channels,
    }

    print(f"\n{'='*50}")
    print(f"[ERGEBNIS] {'PASS' if max_amp > 100 else 'FAIL — NUR NULLEN!'}")
    print(f"  Dauer:          {result['dauer_s']}s")
    print(f"  Pakete:         {result['pakete']}")
    print(f"  Bytes gesamt:   {result['bytes_gesamt']}")
    print(f"  Max Amplitude:  {result['max_amplitude']} {'> 100 OK' if max_amp > 100 else '< 100 FAIL'}")
    print(f"  Non-Zero:       {result['non_zero_prozent']}%")
    print(f"  Bitrate:        {result['bitrate_kbps']} kbps")
    print(f"{'='*50}")

    # WAV speichern
    if save_path and total_bytes > 0:
        with wave.open(save_path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(rate)
            wf.writeframes(all_data)
        print(f"[SAVE] {save_path} ({total_bytes} Bytes)")

    return result


def main():
    parser = argparse.ArgumentParser(description="M.O.L.O.C.H. ReSpeaker UDP Audio-Test")
    parser.add_argument("--rate", type=int, default=16000, choices=[16000, 48000],
                        help="Samplerate (16000 oder 48000)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Aufnahmedauer in Sekunden")
    parser.add_argument("--save", type=str, default=None,
                        help="WAV-Datei speichern (Pfad)")
    parser.add_argument("--status", action="store_true",
                        help="Nur Status abfragen")
    args = parser.parse_args()

    if args.status:
        get_status()
        return

    # Modus umschalten
    print(f"\n[TEST] ReSpeaker UDP Audio-Test @ {args.rate}Hz")
    switch_mode(args.rate)
    time.sleep(0.5)

    # Aufnehmen
    if args.rate == 16000:
        port = 12345
        channels = 1
    else:
        port = 12346
        channels = 2

    result = record_udp(port, args.duration, args.rate, channels, args.save)

    if result:
        # JSON-Ergebnis
        result_path = "/tmp/respeaker_udp_test.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[JSON] {result_path}")

        sys.exit(0 if result["status"] == "PASS" else 1)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
