#!/usr/bin/env python3
"""
Noctua NF-A4x20 5V Luefter-Steuerung via PIO-PWM GPIO18 (Pin 12)
Benoetigt in /boot/firmware/config.txt:
  dtoverlay=pwm-pio,gpio=18   (Pi5/RP1-nativ)

Temperaturkurve (2026-04-10 angepasst):
  Ziel: Sanftere Kurve — Pi5 ist bis 85°C spezifiziert,
  kein Grund bei 46°C schon auf 40-60% zu laufen.
  Noctua laeuft leise im Hintergrund und dreht erst bei echtem Bedarf hoch.

  <50°C   → 25%  (leises Grundrauschen, Minimum)
  50-55°C → 35%  (leicht hochdrehen)
  55-65°C → 50%  (moderate Kuehlung)
  65-75°C → 75%  (aktive Kuehlung)
  >75°C   → 100% (Notfall — alles rein)

  Hysterese: 3°C nach unten (verhindert Oszillation)

Kooperativ: Wenn der Pi5 Active Cooler (cooling_device0) aktiv ist,
  legt der Noctua +15% Duty drauf, um die CPU schneller zu kuehlen.
  Die Erhoehung bleibt bis die Temperatur unter die Hysterese faellt.

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

# Pi5 Active Cooler State (cooling_device0): 0=aus, 1-3=aktiv
CPU_FAN_STATE = "/sys/class/thermal/cooling_device0/cur_state"


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


def cpu_fan_state_lesen():
    """Pi5 Active Cooler State lesen (0=aus, 1-3=aktiv)."""
    try:
        with open(CPU_FAN_STATE) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return 0


def tension_lesen():
    """Liest aktuelle tension aus moloch_status.json (Cross-Process).

    Returns 0.0 wenn Status nicht lesbar / Sentinel < 0 (idle).
    Hardware-als-Ausdruck: hohe Tension -> Noctua hochdrehen, sodass Markus
    den emotionalen Zustand HORT (nicht nur sieht).
    """
    import json
    try:
        with open("/dev/shm/moloch_status.json") as f:
            d = json.load(f)
        t = d.get("tension")
        if isinstance(t, dict):
            t = t.get("level", 0.0)
        elif t is None:
            t = (d.get("core") or {}).get("tension", 0.0)
        t = float(t or 0.0)
        # Sentinel -1.0 = idle, kein Boost
        return max(0.0, t)
    except Exception:
        return 0.0


def tension_zu_duty_boost(tension: float) -> float:
    """Mappt tension [0..1] auf zusaetzlichen Duty-Boost in Prozent.

    >0.7 (Berserker-Naehe) -> +30% Duty
    >0.5 (angespannt)      -> +20%
    >0.3 (aufmerksam)      -> +10%
    sonst                  -> 0%
    """
    if tension >= 0.85: return 0.40   # Maximaler Boost bei kompletter Anspannung
    if tension >= 0.70: return 0.30
    if tension >= 0.50: return 0.20
    if tension >= 0.30: return 0.10
    return 0.0


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

    Sanfte Kurve: Noctua dreht erst ab 50°C hoch — Pi5 ist bis 85°C
    spezifiziert, kein Grund fuer fruehes Hochdrehen.

    Hysterese 3°C: Duty sinkt erst wenn Temp 3°C unter die Schwelle faellt.
    """
    global _letzter_duty_pct
    HYSTERESE = 3.0

    # Schwellen aufsteigend: (Schwelle, Duty%)
    stufen = [
        (75.0, 1.00),   # >75°C  → 100% (Notfall)
        (65.0, 0.75),   # 65-75  → 75%
        (55.0, 0.50),   # 55-65  → 50%
        (50.0, 0.35),   # 50-55  → 35%
    ]

    # Aufwaerts: sofort hochschalten
    for schwelle, duty_pct in stufen:
        if temp >= schwelle:
            _letzter_duty_pct = duty_pct
            return int(PWM_PERIOD_NS * duty_pct)

    # Unter 50°C: Grunddrehzahl, aber mit Hysterese
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

                # Kooperativ: Wenn Pi5 Active Cooler aktiv, Noctua +15%
                cpu_fan = cpu_fan_state_lesen()
                cpu_boost = 0.15 if cpu_fan > 0 else 0.0

                # Hardware-als-Ausdruck: Tension treibt Lüfter hoch.
                # Markus HORT die Anspannung am Noctua. (PC-Topic 2026-05-03)
                tension = tension_lesen()
                tension_boost = tension_zu_duty_boost(tension)

                duty_pct = duty / PWM_PERIOD_NS
                effective_pct = min(1.0, duty_pct + cpu_boost + tension_boost)
                duty = int(PWM_PERIOD_NS * effective_pct)

                tags = []
                if cpu_boost > 0:
                    tags.append(f"CPU-Fan={cpu_fan},+{int(cpu_boost*100)}%")
                if tension_boost > 0:
                    tags.append(f"Tension={tension:.2f},+{int(tension_boost*100)}%")
                tag_str = (" [" + " | ".join(tags) + "]") if tags else ""
                print(f"Temp: {temp:.1f}°C -> Fan: {effective_pct*100:.0f}%{tag_str}",
                      flush=True)

                pwm_schreiben(f"{kanal_pfad}/duty_cycle", duty)
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
