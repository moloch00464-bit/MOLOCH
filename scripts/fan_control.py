#!/usr/bin/env python3
"""
Noctua NF-A4x20 5V Luefter-Steuerung via PIO-PWM GPIO18 (Pin 12)
Benoetigt in /boot/firmware/config.txt:
  dtoverlay=pwm-pio,gpio=18   (Pi5/RP1-nativ)

Temperaturkurve:
  <50°C   → 30%
  50-65°C → 50%
  65-75°C → 75%
  >75°C   → 100%

Fail-Safe: Bei Fehler wird Luefter auf 100% gesetzt.
"""

import os
import time
import subprocess
import sys

# PIO-PWM Konfiguration (Pi5)
# pwm-pio erzeugt einen Chip mit npwm=1, NICHT pwmchip0 (=Pi5-interner Fan, npwm=4)
PWM_CHANNEL = "0"
PWM_PERIOD_NS = 40000  # 25 kHz (Noctua Standard)

POLL_INTERVALL = 5  # Sekunden zwischen Messungen

# Pi5 Built-in Fan Controller hat 4 Kanaele — diesen NICHT verwenden
PI5_BUILTIN_NPWM = 4


def _finde_pwm_chip():
    """
    Findet den von pwm-pio erzeugten Chip.
    Strategie: Alle pwmchipX durchsuchen, Built-in (npwm=4) ueberspringen,
    ersten Chip mit npwm=1 zurueckgeben.
    """
    pwm_basis = "/sys/class/pwm"
    if not os.path.exists(pwm_basis):
        raise RuntimeError("Kein /sys/class/pwm vorhanden — Kernel-PWM nicht geladen?")

    chips = sorted(os.listdir(pwm_basis))  # pwmchip0, pwmchip1, ...
    kandidaten = []

    for chip in chips:
        pfad = f"{pwm_basis}/{chip}"
        try:
            with open(f"{pfad}/npwm") as f:
                npwm = int(f.read().strip())
            if npwm != PI5_BUILTIN_NPWM:
                kandidaten.append((chip, pfad, npwm))
        except OSError:
            continue

    if not kandidaten:
        alle = [c for c in chips]
        raise RuntimeError(
            f"Kein pwm-pio Chip gefunden. Vorhandene Chips: {alle}\n"
            "Ist dtoverlay=pwm-pio,gpio=18 in /boot/firmware/config.txt? Reboot noetig?"
        )

    chip_name, chip_pfad, npwm = kandidaten[0]
    return chip_name, chip_pfad


def pwm_schreiben(pfad, wert):
    with open(pfad, "w") as f:
        f.write(str(wert))


def pwm_init(chip_pfad, kanal_pfad):
    """PWM-Kanal initialisieren."""
    # Export wenn noch nicht exportiert
    if not os.path.exists(kanal_pfad):
        pwm_schreiben(f"{chip_pfad}/export", PWM_CHANNEL)
        time.sleep(0.2)

    # Erst deaktivieren, dann Period setzen (Kernel-Anforderung)
    try:
        pwm_schreiben(f"{kanal_pfad}/enable", 0)
    except OSError:
        pass

    pwm_schreiben(f"{kanal_pfad}/period", PWM_PERIOD_NS)
    pwm_schreiben(f"{kanal_pfad}/duty_cycle", int(PWM_PERIOD_NS * 0.30))
    pwm_schreiben(f"{kanal_pfad}/enable", 1)


def temperatur_lesen():
    """CPU-Temperatur via vcgencmd auslesen."""
    ergebnis = subprocess.run(
        ["vcgencmd", "measure_temp"],
        capture_output=True, text=True, timeout=3
    )
    # Format: temp=52.3'C
    wert = ergebnis.stdout.strip().replace("temp=", "").replace("'C", "")
    return float(wert)


def temp_zu_duty(temp):
    """Temperatur → Duty-Cycle in Nanosekunden."""
    if temp < 50.0:
        return int(PWM_PERIOD_NS * 0.30)   # 30%
    elif temp < 65.0:
        return int(PWM_PERIOD_NS * 0.50)   # 50%
    elif temp < 75.0:
        return int(PWM_PERIOD_NS * 0.75)   # 75%
    else:
        return PWM_PERIOD_NS               # 100%


def main():
    # PWM-Chip finden
    chip_name, chip_pfad = _finde_pwm_chip()
    kanal_pfad = f"{chip_pfad}/pwm{PWM_CHANNEL}"

    print(f"Fan Control: {chip_name}/pwm{PWM_CHANNEL}, GPIO18, 25kHz", flush=True)

    try:
        pwm_init(chip_pfad, kanal_pfad)
        print("PWM initialisiert. Starte Temperatur-Regelschleife.", flush=True)

        while True:
            try:
                temp = temperatur_lesen()
                duty = temp_zu_duty(temp)
                pwm_schreiben(f"{kanal_pfad}/duty_cycle", duty)
                pct = (duty / PWM_PERIOD_NS) * 100
                print(f"Temp: {temp:.1f}°C → Fan: {pct:.0f}%", flush=True)
            except Exception as e:
                # Messfehler: Luefter sicher auf 100% setzen
                print(f"WARNUNG Messfehler: {e} → Luefter auf 100%", file=sys.stderr, flush=True)
                try:
                    pwm_schreiben(f"{kanal_pfad}/duty_cycle", PWM_PERIOD_NS)
                except Exception:
                    pass

            time.sleep(POLL_INTERVALL)

    except KeyboardInterrupt:
        # Sauberes Beenden: Luefter auf 100% (Sicherheit)
        try:
            pwm_schreiben(f"{kanal_pfad}/duty_cycle", PWM_PERIOD_NS)
        except Exception:
            pass
        print("Beendet. Luefter auf 100% gesetzt.", flush=True)

    except Exception as e:
        print(f"FEHLER: {e}", file=sys.stderr, flush=True)
        # Fail-Safe: Luefter auf 100%
        try:
            pwm_schreiben(f"{kanal_pfad}/duty_cycle", PWM_PERIOD_NS)
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
