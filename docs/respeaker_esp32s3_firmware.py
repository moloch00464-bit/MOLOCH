"""
M.O.L.O.C.H. — ReSpeaker Lite ESP32-S3 WiFi-Mic Firmware
=========================================================

MicroPython-Script fuer den ESP32-S3 auf dem ReSpeaker Lite Board.
Liest I2S Audio vom XMOS XU316 und streamt per TCP an den Pi.

Funktionen:
- I2S vom XMOS lesen (48kHz, Stereo, 32-bit)
- Ringpuffer 4096 Samples
- Decimation 3:1 → 16kHz Mono Stream
- TCP-Server Port 12345: 16kHz/16-bit/Mono (fuer Whisper)
- TCP-Server Port 12346: 48kHz/16-bit/Stereo (fuer Stimmbiometrie)
- TCP-Client Port 12347: Empfaengt Audio vom Pi → I2S raus (Lautsprecher)
- WLAN-Verlust: Weiter aufnehmen, bei Reconnect Puffer nachsenden
- RGB-LED (WS2812 auf GPIO 1): Statusanzeige + UDP-Steuerung Port 8888
- LED-Kommandos: "LED:farbe [modus] [geschwindigkeit]" per UDP vom Pi

I2S Pin-Belegung ReSpeaker Lite (ESP32-S3 ↔ XMOS):
- SCK (BCLK):  GPIO 8
- WS  (LRCK):  GPIO 7
- SDI (DIN, Daten rein von XMOS): GPIO 44
- SDO (DOUT, Daten raus zu XMOS): GPIO 43

Quelle: Seeed Studio Wiki + GitHub respeaker/ReSpeaker_Lite

Deployment:
  1. esptool.py --port /dev/ttyACM0 erase_flash
  2. esptool.py --port /dev/ttyACM0 --baud 460800 write_flash -z 0x0 ESP32_GENERIC_S3-xxxxx.bin
  3. mpremote connect /dev/ttyACM0 cp docs/respeaker_esp32s3_firmware.py :main.py
  4. mpremote connect /dev/ttyACM0 reset

Author: M.O.L.O.C.H. System
"""

import machine
import network
import socket
import time
import struct
import gc
import neopixel
from machine import I2S, Pin

# =============================================================================
# Konfiguration
# =============================================================================

# WLAN-Credentials — Direkt-Verbindung zum Pi5 AP (kein Router)
WIFI_SSID = "MOLOCH_DIRECT"
WIFI_PASS = "M0l0ch2026!"

# Statische IP im Pi-AP Subnet (Pi = 10.42.0.1)
STATIC_IP = "10.42.0.2"
SUBNET_MASK = "255.255.255.0"
GATEWAY = "10.42.0.1"
DNS = "10.42.0.1"

# I2S Pins (ReSpeaker Lite ESP32-S3 → XMOS XU316)
# Quelle: Seeed Studio Wiki + GitHub respeaker/ReSpeaker_Lite
I2S_SCK_PIN = 8    # BCLK
I2S_WS_PIN = 7     # LRCK (Word Select)
I2S_SDI_PIN = 44   # DIN — Serial Data In (von XMOS, Mikrofon-Daten)
I2S_SDO_PIN = 43   # DOUT — Serial Data Out (zu XMOS, Lautsprecher-Daten)

# Audio-Parameter
SAMPLE_RATE_48K = 48000
SAMPLE_RATE_16K = 16000
CHANNELS = 2       # Stereo vom XMOS
BITS = 32          # XMOS liefert 32-bit I2S
DECIMATION = 3     # 48kHz / 3 = 16kHz

# Ringpuffer
RING_BUFFER_SAMPLES = 4096
RING_BUFFER_BYTES = RING_BUFFER_SAMPLES * CHANNELS * 4  # 32-bit = 4 Bytes

# TCP Ports
PORT_16K = 12345   # 16kHz Mono Stream (Whisper)
PORT_48K = 12346   # 48kHz Stereo Stream (Stimmbiometrie)
PORT_SPK = 12347   # Lautsprecher-Empfang vom Pi

# I2S DMA Buffer
DMA_BUF_LEN = 1024
DMA_BUF_COUNT = 4

# Offline-Puffer bei WLAN-Verlust (max 10s bei 16kHz Mono 16-bit)
OFFLINE_BUFFER_MAX = SAMPLE_RATE_16K * 2 * 10  # 320.000 Bytes

# RGB-LED (WS2812 NeoPixel)
LED_PIN = 1            # GPIO 1 auf ReSpeaker Lite
LED_BRIGHTNESS = 50    # Max Helligkeit (0-255), nicht blenden
LED_UDP_PORT = 8888    # UDP-Port fuer LED-Kommandos vom Pi

# =============================================================================
# Globaler State
# =============================================================================

wlan = None
i2s_in = None
i2s_out = None
clients_16k = []
clients_48k = []
offline_buffer = bytearray()

# LED State
np = neopixel.NeoPixel(Pin(LED_PIN), 1)
led_color = (0, 0, 0)       # Aktuelle Farbe (R, G, B)
led_mode = "statisch"        # statisch, pulsierend, blinkend, regenbogen
led_speed = 500              # Animations-Intervall in ms
led_phase = 0                # Animations-Phase
led_last_update = 0          # Letzter Update-Zeitpunkt (ms)
udp_led = None               # UDP-Socket fuer LED-Kommandos


# =============================================================================
# WLAN
# =============================================================================

def wifi_connect():
    """Direkt-WLAN zum Pi5 AP verbinden, statische IP."""
    global wlan
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)

    # Statische IP setzen (kein DHCP noetig)
    wlan.ifconfig((STATIC_IP, SUBNET_MASK, GATEWAY, DNS))

    if wlan.isconnected():
        print(f"[WIFI] Bereits verbunden: {wlan.ifconfig()[0]}")
        return True

    print(f"[WIFI] Verbinde mit Pi5 AP '{WIFI_SSID}'...")
    wlan.connect(WIFI_SSID, WIFI_PASS)

    timeout = 15
    while not wlan.isconnected() and timeout > 0:
        time.sleep(1)
        timeout -= 1

    if wlan.isconnected():
        ip = wlan.ifconfig()[0]
        print(f"[WIFI] Verbunden! IP: {ip} (Pi5 = {GATEWAY})")
        return True
    else:
        print("[WIFI] Verbindung fehlgeschlagen!")
        return False


def wifi_check():
    """Prueft WLAN, reconnected bei Bedarf. Gibt True zurueck wenn online."""
    global wlan
    if wlan and wlan.isconnected():
        return True

    print("[WIFI] Verbindung verloren, versuche Reconnect...")
    return wifi_connect()


# =============================================================================
# I2S Setup
# =============================================================================

def i2s_setup():
    """I2S Eingang (Mikrofon vom XMOS) und Ausgang (Lautsprecher) initialisieren."""
    global i2s_in, i2s_out

    # I2S Eingang: XMOS → ESP32 (Mikrofon-Daten)
    i2s_in = I2S(
        0,
        sck=Pin(I2S_SCK_PIN),
        ws=Pin(I2S_WS_PIN),
        sd=Pin(I2S_SDI_PIN),
        mode=I2S.RX,
        bits=BITS,
        format=I2S.STEREO,
        rate=SAMPLE_RATE_48K,
        ibuf=DMA_BUF_LEN * DMA_BUF_COUNT * CHANNELS * 4
    )

    # I2S Ausgang: ESP32 → XMOS (Lautsprecher-Daten)
    i2s_out = I2S(
        1,
        sck=Pin(I2S_SCK_PIN),
        ws=Pin(I2S_WS_PIN),
        sd=Pin(I2S_SDO_PIN),
        mode=I2S.TX,
        bits=BITS,
        format=I2S.STEREO,
        rate=SAMPLE_RATE_48K,
        ibuf=DMA_BUF_LEN * DMA_BUF_COUNT * CHANNELS * 4
    )

    print("[I2S] Eingang + Ausgang initialisiert (48kHz Stereo 32-bit)")


# =============================================================================
# Decimation: 48kHz Stereo 32-bit → 16kHz Mono 16-bit
# =============================================================================

def decimate_48k_to_16k(buf_48k, num_bytes):
    """
    48kHz Stereo 32-bit → 16kHz Mono 16-bit.
    Nimmt jeden 3. Frame, mittelt L+R, schneidet auf 16-bit.
    """
    # Jeder Frame = 8 Bytes (2 Kanäle * 4 Bytes)
    frame_size = CHANNELS * 4
    num_frames = num_bytes // frame_size
    out_frames = num_frames // DECIMATION

    if out_frames == 0:
        return bytes(0)

    out = bytearray(out_frames * 2)  # 16-bit Mono = 2 Bytes pro Sample
    out_idx = 0

    for i in range(0, num_frames, DECIMATION):
        offset = i * frame_size
        # Links (32-bit signed, little-endian)
        left = struct.unpack_from('<i', buf_48k, offset)[0]
        # Rechts (32-bit signed, little-endian)
        right = struct.unpack_from('<i', buf_48k, offset + 4)[0]
        # Mittelwert, auf 16-bit skalieren (obere 16 Bit nehmen)
        mono = (left + right) // 2
        sample_16 = mono >> 16  # 32-bit → 16-bit
        # Clamp
        if sample_16 > 32767:
            sample_16 = 32767
        elif sample_16 < -32768:
            sample_16 = -32768
        struct.pack_into('<h', out, out_idx, sample_16)
        out_idx += 2

    return bytes(out[:out_idx])


def convert_48k_to_16bit(buf_48k, num_bytes):
    """
    48kHz Stereo 32-bit → 48kHz Stereo 16-bit.
    Obere 16 Bit jedes 32-bit Samples nehmen.
    """
    num_samples = num_bytes // 4  # Jedes 32-bit Sample = 4 Bytes
    out = bytearray(num_samples * 2)

    for i in range(num_samples):
        val = struct.unpack_from('<i', buf_48k, i * 4)[0]
        sample_16 = val >> 16
        if sample_16 > 32767:
            sample_16 = 32767
        elif sample_16 < -32768:
            sample_16 = -32768
        struct.pack_into('<h', out, i * 2, sample_16)

    return bytes(out)


# =============================================================================
# TCP Server
# =============================================================================

def tcp_server_setup(port):
    """Nicht-blockierenden TCP-Server auf Port starten."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', port))
    srv.listen(2)
    srv.setblocking(False)
    print(f"[TCP] Server lauscht auf Port {port}")
    return srv


def tcp_accept(srv, client_list, label):
    """Neue Clients akzeptieren (non-blocking)."""
    try:
        client, addr = srv.accept()
        client.setblocking(False)
        client_list.append(client)
        print(f"[TCP] {label}: Client verbunden von {addr}")
    except OSError:
        pass  # Kein neuer Client


def tcp_send_all(client_list, data):
    """Daten an alle verbundenen Clients senden, tote entfernen."""
    dead = []
    for c in client_list:
        try:
            c.sendall(data)
        except OSError:
            dead.append(c)

    for c in dead:
        try:
            c.close()
        except:
            pass
        client_list.remove(c)
        print("[TCP] Client getrennt")


# =============================================================================
# Lautsprecher-Empfang (Port 12347)
# =============================================================================

def speaker_receive_loop(srv_spk):
    """Empfaengt Audio-Daten vom Pi und gibt sie per I2S aus."""
    global i2s_out
    spk_client = None

    try:
        client, addr = srv_spk.accept()
        client.setblocking(False)
        spk_client = client
        print(f"[SPK] Pi verbunden von {addr}")
    except OSError:
        pass

    if spk_client:
        try:
            data = spk_client.recv(4096)
            if data and i2s_out:
                # Pi sendet 48kHz Stereo 16-bit, wir brauchen 32-bit fuer I2S
                # Upscale 16-bit → 32-bit
                num_samples = len(data) // 2
                out = bytearray(num_samples * 4)
                for i in range(num_samples):
                    val = struct.unpack_from('<h', data, i * 2)[0]
                    struct.pack_into('<i', out, i * 4, val << 16)
                i2s_out.write(out)
        except OSError:
            pass

    return spk_client


# =============================================================================
# Offline-Puffer (bei WLAN-Verlust)
# =============================================================================

def offline_buffer_add(data_16k):
    """Audio im Offline-Puffer speichern wenn WLAN weg."""
    global offline_buffer
    if len(offline_buffer) + len(data_16k) <= OFFLINE_BUFFER_MAX:
        offline_buffer.extend(data_16k)


def offline_buffer_flush():
    """Gespeicherte Daten nach Reconnect an Clients senden."""
    global offline_buffer
    if len(offline_buffer) > 0 and len(clients_16k) > 0:
        print(f"[OFFLINE] Sende {len(offline_buffer)} Bytes gepuffertes Audio...")
        # In Chunks senden um Speicher zu schonen
        chunk = 4096
        for i in range(0, len(offline_buffer), chunk):
            tcp_send_all(clients_16k, offline_buffer[i:i+chunk])
        offline_buffer = bytearray()
        print("[OFFLINE] Puffer geleert")


# =============================================================================
# RGB-LED (WS2812 NeoPixel auf GPIO 1)
# =============================================================================

# Farb-Tabelle (GRB-Order wird von neopixel intern gehandelt)
FARBEN = {
    "rot":     (255, 0, 0),
    "gruen":   (0, 255, 0),
    "blau":    (0, 0, 255),
    "gelb":    (255, 180, 0),
    "cyan":    (0, 255, 255),
    "magenta": (255, 0, 255),
    "weiss":   (255, 255, 255),
    "aus":     (0, 0, 0),
}

GESCHWINDIGKEIT = {
    "langsam": 1000,
    "mittel":  500,
    "schnell": 150,
}


def led_set(r, g, b):
    """LED auf Farbe setzen (mit Helligkeitsbegrenzung)."""
    scale = LED_BRIGHTNESS / 255.0
    np[0] = (int(r * scale), int(g * scale), int(b * scale))
    np.write()


def led_test():
    """Startup-Test: Rot → Gruen → Blau → Aus."""
    print("[LED] Testroutine...")
    for farbe in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]:
        led_set(*farbe)
        time.sleep_ms(400)
    led_set(0, 0, 0)
    print("[LED] Test OK")


def led_update():
    """LED-Animation aktualisieren (non-blocking, via millis)."""
    global led_phase, led_last_update

    now = time.ticks_ms()
    if time.ticks_diff(now, led_last_update) < (led_speed // 4 if led_mode == "blinkend" else led_speed // 10):
        return

    led_last_update = now
    r, g, b = led_color

    if led_mode == "statisch":
        led_set(r, g, b)
        return

    elif led_mode == "pulsierend":
        # Sinus-Approximation: 0..1..0 ueber 20 Schritte
        led_phase = (led_phase + 1) % 20
        if led_phase < 10:
            brightness = led_phase / 10.0
        else:
            brightness = (20 - led_phase) / 10.0
        # Minimum-Helligkeit damit LED nie ganz aus ist
        brightness = 0.05 + brightness * 0.95
        led_set(int(r * brightness), int(g * brightness), int(b * brightness))

    elif led_mode == "blinkend":
        led_phase = (led_phase + 1) % 2
        if led_phase == 0:
            led_set(r, g, b)
        else:
            led_set(0, 0, 0)

    elif led_mode == "regenbogen":
        led_phase = (led_phase + 1) % 60
        # HSV-artiger Regenbogen ueber 60 Schritte
        pos = led_phase
        if pos < 10:
            led_set(255, pos * 25, 0)
        elif pos < 20:
            led_set(255 - (pos - 10) * 25, 255, 0)
        elif pos < 30:
            led_set(0, 255, (pos - 20) * 25)
        elif pos < 40:
            led_set(0, 255 - (pos - 30) * 25, 255)
        elif pos < 50:
            led_set((pos - 40) * 25, 0, 255)
        else:
            led_set(255, 0, 255 - (pos - 50) * 25)


def led_parse_command(cmd_str):
    """
    LED-Kommando parsen.
    Format: "LED:farbe [modus] [geschwindigkeit]"
    Beispiel: "LED:rot pulsierend schnell"
    """
    global led_color, led_mode, led_speed, led_phase

    cmd_str = cmd_str.strip().lower()
    if not cmd_str.startswith("led:"):
        return

    parts = cmd_str[4:].split()
    if not parts:
        return

    # Farbe
    farbe_name = parts[0]
    if farbe_name in FARBEN:
        led_color = FARBEN[farbe_name]
    else:
        # Hex-Farbe versuchen: "ff00aa"
        try:
            if len(farbe_name) == 6:
                led_color = (int(farbe_name[0:2], 16),
                             int(farbe_name[2:4], 16),
                             int(farbe_name[4:6], 16))
        except ValueError:
            print(f"[LED] Unbekannte Farbe: {farbe_name}")
            return

    # Modus
    if len(parts) > 1 and parts[1] in ("statisch", "pulsierend", "blinkend", "regenbogen"):
        led_mode = parts[1]
    else:
        led_mode = "statisch"

    # Geschwindigkeit
    if len(parts) > 2 and parts[2] in GESCHWINDIGKEIT:
        led_speed = GESCHWINDIGKEIT[parts[2]]

    led_phase = 0
    print(f"[LED] {farbe_name} {led_mode} {led_speed}ms")

    # Sofort setzen bei statisch
    if led_mode == "statisch":
        led_set(*led_color)


def udp_led_setup():
    """UDP-Server fuer LED-Kommandos starten."""
    global udp_led
    udp_led = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_led.bind(('0.0.0.0', LED_UDP_PORT))
    udp_led.setblocking(False)
    print(f"[LED] UDP-Server auf Port {LED_UDP_PORT}")


def udp_led_check():
    """Prueft ob LED-Kommando eingegangen ist (non-blocking)."""
    global udp_led
    if not udp_led:
        return
    try:
        data, addr = udp_led.recvfrom(256)
        if data:
            cmd = data.decode('utf-8', 'ignore')
            led_parse_command(cmd)
    except OSError:
        pass  # Kein Paket da


# =============================================================================
# Hauptschleife
# =============================================================================

def main():
    global clients_16k, clients_48k, offline_buffer

    print("=" * 50)
    print("M.O.L.O.C.H. ReSpeaker WiFi-Mic + RGB-LED")
    print("ESP32-S3 + XMOS XU316 I2S Bridge")
    print("=" * 50)

    # LED-Test zuerst (zeigt dass ESP32 lebt)
    led_test()

    # I2S initialisieren (laeuft auch ohne WLAN)
    i2s_setup()

    # LED: Blau pulsierend waehrend WLAN-Verbindung
    led_parse_command("LED:blau pulsierend mittel")

    # WLAN verbinden
    wifi_ok = wifi_connect()

    # TCP Server starten (nur wenn WLAN da)
    srv_16k = None
    srv_48k = None
    srv_spk = None

    if wifi_ok:
        srv_16k = tcp_server_setup(PORT_16K)
        srv_48k = tcp_server_setup(PORT_48K)
        srv_spk = tcp_server_setup(PORT_SPK)
        udp_led_setup()
        # LED: Gruen kurz aufblitzen = verbunden
        led_set(0, 255, 0)
        time.sleep_ms(500)
        # Dann blau pulsierend langsam = idle
        led_parse_command("LED:blau pulsierend langsam")
    else:
        # LED: Rot blinkend = kein WLAN
        led_parse_command("LED:rot blinkend schnell")

    # I2S Lese-Puffer
    read_buf = bytearray(DMA_BUF_LEN * CHANNELS * 4)  # 32-bit Stereo
    wifi_check_counter = 0

    print("[MAIN] Starte Audio-Loop...")

    while True:
        try:
            # === I2S Lesen ===
            num_read = i2s_in.readinto(read_buf)
            if num_read == 0:
                time.sleep_ms(1)
                continue

            # === Decimation und Konvertierung ===
            data_16k = decimate_48k_to_16k(read_buf, num_read)
            data_48k = convert_48k_to_16bit(read_buf, num_read)

            # === WLAN-Status pruefen (alle ~500 Iterationen) ===
            wifi_check_counter += 1
            if wifi_check_counter >= 500:
                wifi_check_counter = 0
                was_offline = not (wlan and wlan.isconnected())

                if not wifi_check():
                    # Offline: Audio puffern
                    offline_buffer_add(data_16k)
                    continue
                elif was_offline:
                    # Gerade reconnected: Server neu starten
                    print("[MAIN] WLAN wieder da!")
                    if srv_16k is None:
                        srv_16k = tcp_server_setup(PORT_16K)
                        srv_48k = tcp_server_setup(PORT_48K)
                        srv_spk = tcp_server_setup(PORT_SPK)
                    # Gepufferte Daten senden
                    offline_buffer_flush()

            # === Neue TCP-Clients akzeptieren ===
            if srv_16k:
                tcp_accept(srv_16k, clients_16k, "16kHz")
            if srv_48k:
                tcp_accept(srv_48k, clients_48k, "48kHz")

            # === Audio an Clients streamen ===
            if len(clients_16k) > 0:
                tcp_send_all(clients_16k, data_16k)
            elif not (wlan and wlan.isconnected()):
                offline_buffer_add(data_16k)

            if len(clients_48k) > 0:
                tcp_send_all(clients_48k, data_48k)

            # === Lautsprecher-Empfang ===
            if srv_spk:
                speaker_receive_loop(srv_spk)

            # === LED Animation + UDP Kommandos ===
            led_update()
            udp_led_check()

            # === GC alle paar Zyklen ===
            if wifi_check_counter % 100 == 0:
                gc.collect()

        except KeyboardInterrupt:
            print("[MAIN] Beende...")
            break
        except Exception as e:
            print(f"[MAIN] Fehler: {e}")
            time.sleep_ms(100)

    # Aufraeumen
    if i2s_in:
        i2s_in.deinit()
    if i2s_out:
        i2s_out.deinit()
    for c in clients_16k + clients_48k:
        try:
            c.close()
        except:
            pass
    print("[MAIN] Beendet.")


# =============================================================================
# Start
# =============================================================================
if __name__ == "__main__":
    main()
