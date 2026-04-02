#!/usr/bin/env python3
"""
Noctua NF-A4x20 5V Luefter-Steuerung via PIO-PWM GPIO18 (Pin 12)
Benoetigt in /boot/firmware/config.txt:
  dtoverlay=pwm-pio,gpio=18   (Pi5/RP1-nativ)

Temperaturkurve (2026-04-02 angepasst):
  Ziel: Noctua dreht FRUEHER hoch, damit der Pi5 Active Cooler
  (cooling_device0, Level 1 ab 50°C) gar nicht anspringen muss.
  Der Noctua soll die CPU unter 50°C halten.

  <42°C   → 25%  (leises Grundrauschen)
  42-47°C → 40%  (praeventiv hochdrehen)
  47-52°C → 60%  (CPU-Kuehler Schwelle naht)
  52-60°C → 80%  (CPU-Kuehler sollte jetzt schon wieder ausgehen)
  >60°C   → 100% (Notfall — alles rein)

  Hysterese: 3°C nach unten (verhindert Oszillation)

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
    pwm_schreiben(f"{kanal_pfad}/duty_cycle", int(PWM_PERIOD_NS * 0.25))
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


# Hysterese-Zustand: letzter Duty-Wert merken
_letzter_duty_pct = 0.25


def temp_zu_duty(temp):
    """
    Temperatur → Duty-Cycle in Nanosekunden.

    Aggressive Kurve: Noctua dreht ab 42°C hoch, damit der Pi5 Active
    Cooler (cooling_device0 Level 1 ab 50°C) nicht anspringen muss.

    Hysterese 3°C: Duty sinkt erst wenn Temp 3°C unter die Schwelle faellt.
    """
    global _letzter_duty_pct
    HYSTERESE = 3.0

    # Schwellen aufsteigend: (Schwelle, Duty%)
    stufen = [
        (60.0, 1.00),   # >60°C  → 100%
        (52.0, 0.80),   # 52-60  → 80%
        (47.0, 0.60),   # 47-52  → 60%
        (42.0, 0.40),   # 42-47  → 40%
    ]

    # Aufwaerts: sofort hochschalten
    for schwelle, duty_pct in stufen:
        if temp >= schwelle:
            _letzter_duty_pct = duty_pct
            return int(PWM_PERIOD_NS * duty_pct)

    # Unter 42°C: Grunddrehzahl, aber mit Hysterese
    # Nur runterschalten wenn Temp 3°C unter der letzten Schwelle liegt
    if _letzter_duty_pct > 0.25:
        # Finde die Schwelle die zum letzten Duty gehoert
        for schwelle, duty_pct in stufen:
            if duty_pct == _letzter_duty_pct:
                if temp >= (schwelle - HYSTERESE):
                    # Noch nicht kalt genug — bleib auf aktuellem Level
                    return int(PWM_PERIOD_NS * _letzter_duty_pct)
                break

    _letzter_duty_pct = 0.25
    return int(PWM_PERIOD_NS * 0.25)   # 25% Grunddrehzahl


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
