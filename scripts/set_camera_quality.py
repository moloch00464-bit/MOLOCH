#!/usr/bin/env python3
"""
Sonoff CAM-PT2 ONVIF Qualitaets-Optimierung
=============================================

Setzt VideoEncoder-Parameter auf maximale Qualitaet:
- Quality: 2 -> 5 (Maximum)
- BitrateLimit: 1500 -> 4000 kbit/s
- GovLength (GOP): 80 -> 40 (kuerzere GOPs = bessere Qualitaet)

Laeuft parallel zum RTSP-Stream (ONVIF nutzt separaten Port 80).
"""

import os
import sys
import logging
from onvif import ONVIFCamera
from zeep.helpers import serialize_object

# Kamera-Credentials aus Umgebungsvariablen
CAMERA_IP = os.environ.get("MOLOCH_CAMERA_HOST", "192.168.178.25")
USERNAME = os.environ.get("MOLOCH_CAMERA_USER", "Moloch_4.5")
PASSWORD = os.environ.get("MOLOCH_CAMERA_PASS", "Auge666")
ONVIF_PORT = 80

# Zielwerte
TARGET_QUALITY = 5       # Max laut ONVIF (vorher: 2)
TARGET_BITRATE = 4000    # kbit/s (vorher: 1500)
TARGET_GOP = 40          # GovLength (vorher: 80, Range Min=Max=40)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    print("=" * 60)
    print("SONOFF CAM-PT2 — ONVIF Qualitaets-Optimierung")
    print("=" * 60)

    # 1. Verbindung herstellen
    print(f"\n[1] Verbinde mit {CAMERA_IP}:{ONVIF_PORT} ...")
    try:
        camera = ONVIFCamera(CAMERA_IP, ONVIF_PORT, USERNAME, PASSWORD)
    except Exception as e:
        logger.error(f"ONVIF-Verbindung fehlgeschlagen: {e}")
        sys.exit(1)
    print("    Verbunden.")

    # 2. Media Service + Profile holen
    print("\n[2] Media-Profile laden ...")
    media = camera.create_media_service()
    profiles = media.GetProfiles()
    if not profiles:
        logger.error("Keine Media-Profile gefunden!")
        sys.exit(1)

    profile = profiles[0]
    print(f"    Profil: {profile.Name} (Token: {profile.token})")

    # 3. Aktuelle VideoEncoder Config lesen
    print("\n[3] Aktuelle VideoEncoder-Config:")
    enc = profile.VideoEncoderConfiguration
    if not enc:
        logger.error("Kein VideoEncoderConfiguration im Profil!")
        sys.exit(1)

    print(f"    Token:      {enc.token}")
    print(f"    Encoding:   {enc.Encoding}")
    print(f"    Resolution: {enc.Resolution.Width}x{enc.Resolution.Height}")
    print(f"    Quality:    {enc.Quality}")
    print(f"    FrameRate:  {enc.RateControl.FrameRateLimit}")
    print(f"    Bitrate:    {enc.RateControl.BitrateLimit} kbit/s")
    if hasattr(enc, 'H264') and enc.H264:
        print(f"    GovLength:  {enc.H264.GovLength}")
        print(f"    H264Profile: {enc.H264.H264Profile}")

    # 4. VideoEncoder Config Options holen (Ranges pruefen)
    print("\n[4] Erlaubte Ranges pruefen ...")
    try:
        options = media.GetVideoEncoderConfigurationOptions({
            'ConfigurationToken': enc.token,
            'ProfileToken': profile.token
        })
        if hasattr(options, 'QualityRange'):
            qr = options.QualityRange
            print(f"    QualityRange: {qr.Min} - {qr.Max}")
        if hasattr(options, 'H264') and options.H264:
            h264_opts = options.H264
            if hasattr(h264_opts, 'GovLengthRange'):
                gr = h264_opts.GovLengthRange
                print(f"    GovLengthRange: {gr.Min} - {gr.Max}")
            if hasattr(h264_opts, 'FrameRateRange'):
                fr = h264_opts.FrameRateRange
                print(f"    FrameRateRange: {fr.Min} - {fr.Max}")
            if hasattr(h264_opts, 'EncodingIntervalRange'):
                ei = h264_opts.EncodingIntervalRange
                print(f"    EncodingIntervalRange: {ei.Min} - {ei.Max}")
    except Exception as e:
        logger.warning(f"    Options konnten nicht gelesen werden: {e}")

    # 5. Neue Werte setzen
    print(f"\n[5] Setze neue Werte:")
    print(f"    Quality:    {enc.Quality} -> {TARGET_QUALITY}")
    print(f"    Bitrate:    {enc.RateControl.BitrateLimit} -> {TARGET_BITRATE} kbit/s")

    enc.Quality = TARGET_QUALITY
    enc.RateControl.BitrateLimit = TARGET_BITRATE

    if hasattr(enc, 'H264') and enc.H264:
        old_gov = enc.H264.GovLength
        enc.H264.GovLength = TARGET_GOP
        print(f"    GovLength:  {old_gov} -> {TARGET_GOP}")

    # 6. Config zurueckschreiben
    print("\n[6] Schreibe Config zurueck ...")
    try:
        media.SetVideoEncoderConfiguration({
            'Configuration': enc,
            'ForcePersistence': True
        })
        print("    ERFOLG — Config geschrieben!")
    except Exception as e:
        logger.error(f"    SetVideoEncoderConfiguration fehlgeschlagen: {e}")
        print("    Hinweis: Manche Sonoff-Kameras ignorieren bestimmte Parameter.")
        print("    Versuche nur Quality zu setzen ...")

        # Fallback: nur Quality aendern, Bitrate/GOP zuruecksetzen
        try:
            # Nochmal frisch laden
            profiles2 = media.GetProfiles()
            enc2 = profiles2[0].VideoEncoderConfiguration
            enc2.Quality = TARGET_QUALITY
            media.SetVideoEncoderConfiguration({
                'Configuration': enc2,
                'ForcePersistence': True
            })
            print("    Fallback ERFOLG — Quality gesetzt!")
        except Exception as e2:
            logger.error(f"    Auch Fallback fehlgeschlagen: {e2}")
            sys.exit(1)

    # 7. Verifizieren
    print("\n[7] Verifiziere ...")
    profiles_after = media.GetProfiles()
    enc_after = profiles_after[0].VideoEncoderConfiguration
    print(f"    Quality:    {enc_after.Quality}")
    print(f"    Bitrate:    {enc_after.RateControl.BitrateLimit} kbit/s")
    if hasattr(enc_after, 'H264') and enc_after.H264:
        print(f"    GovLength:  {enc_after.H264.GovLength}")

    # Erfolg pruefen
    ok = True
    if enc_after.Quality != TARGET_QUALITY:
        print(f"    WARNUNG: Quality ist {enc_after.Quality}, nicht {TARGET_QUALITY}")
        ok = False
    if enc_after.RateControl.BitrateLimit < 2000:
        print(f"    WARNUNG: Bitrate nur {enc_after.RateControl.BitrateLimit} kbit/s")

    if ok:
        print("\n" + "=" * 60)
        print("FERTIG — Kamera-Qualitaet optimiert!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("TEILWEISE — einige Parameter wurden moeglicherweise ignoriert")
        print("=" * 60)


if __name__ == "__main__":
    main()
