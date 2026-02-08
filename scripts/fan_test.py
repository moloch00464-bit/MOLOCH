#!/usr/bin/env python3
"""
M.O.L.O.C.H. Fan Test - Automatischer Lüfter-Stresstest
"""

import subprocess
import time
import sys

# ANSI Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

def get_temp():
    """Get CPU temperature."""
    try:
        result = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True)
        temp_str = result.stdout.strip().replace("temp=", "").replace("'C", "")
        return float(temp_str)
    except:
        return 0.0

def get_fan_state():
    """Get fan PWM state (0-3)."""
    try:
        with open("/sys/class/thermal/cooling_device0/cur_state") as f:
            return int(f.read().strip())
    except:
        return -1

def get_fan_speed_text(state):
    """Convert fan state to readable text."""
    speeds = {-1: "NICHT ERKANNT", 0: "AUS", 1: "NIEDRIG", 2: "MITTEL", 3: "HOCH"}
    return speeds.get(state, f"UNBEKANNT ({state})")

def temp_color(temp):
    """Get color based on temperature."""
    if temp < 50:
        return GREEN
    elif temp < 65:
        return YELLOW
    else:
        return RED

def main():
    print(f"""
{CYAN}╔══════════════════════════════════════════════════════════════╗
║           M.O.L.O.C.H. LÜFTER STRESSTEST                     ║
╚══════════════════════════════════════════════════════════════╝{RESET}
""")

    # Initial status
    temp = get_temp()
    fan = get_fan_state()

    print(f"{BOLD}INITIAL STATUS:{RESET}")
    print(f"  Temperatur: {temp_color(temp)}{temp:.1f}°C{RESET}")
    print(f"  Lüfter:     {get_fan_speed_text(fan)}")
    print()

    if fan == -1:
        print(f"{RED}FEHLER: Lüfter-Device nicht gefunden!{RESET}")
        print("Prüfe ob Active Cooler angeschlossen ist.")
        print("Config in /boot/firmware/config.txt prüfen.")
        sys.exit(1)

    # Start stress test
    print(f"{YELLOW}Starte CPU Stress-Test (60 Sekunden)...{RESET}")
    print(f"{CYAN}Beobachte ob der Lüfter anfängt zu drehen!{RESET}")
    print()
    print(f"{'Zeit':>6} │ {'Temp':>8} │ {'Lüfter':>12} │ Status")
    print("─" * 50)

    # Start stress in background
    stress_proc = subprocess.Popen(
        ["stress", "--cpu", "4", "--timeout", "60s"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    fan_activated = False
    max_temp = temp
    start_time = time.time()

    try:
        for i in range(13):  # 13 x 5s = 65s
            time.sleep(5)

            elapsed = int(time.time() - start_time)
            temp = get_temp()
            fan = get_fan_state()
            max_temp = max(max_temp, temp)

            # Status indicator
            if fan > 0 and not fan_activated:
                fan_activated = True
                status = f"{GREEN}>>> LÜFTER GESTARTET! <<<{RESET}"
            elif fan > 0:
                status = f"{GREEN}AKTIV{RESET}"
            elif temp > 60:
                status = f"{YELLOW}AUFHEIZEND...{RESET}"
            else:
                status = ""

            print(f"{elapsed:>5}s │ {temp_color(temp)}{temp:>6.1f}°C{RESET} │ {get_fan_speed_text(fan):>12} │ {status}")

            # Check if stress finished
            if stress_proc.poll() is not None:
                break

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Abgebrochen.{RESET}")
        stress_proc.terminate()

    # Wait for stress to finish
    stress_proc.wait()

    # Cooldown phase
    print()
    print(f"{CYAN}Abkühlphase (30 Sekunden)...{RESET}")
    print(f"{'Zeit':>6} │ {'Temp':>8} │ {'Lüfter':>12}")
    print("─" * 35)

    for i in range(6):  # 6 x 5s = 30s
        time.sleep(5)
        temp = get_temp()
        fan = get_fan_state()
        print(f"{(i+1)*5:>5}s │ {temp_color(temp)}{temp:>6.1f}°C{RESET} │ {get_fan_speed_text(fan):>12}")

    # Final result
    print()
    print(f"{CYAN}╔══════════════════════════════════════════════════════════════╗")
    print(f"║                      ERGEBNIS                                ║")
    print(f"╚══════════════════════════════════════════════════════════════╝{RESET}")
    print()

    if fan_activated:
        print(f"  {GREEN}{BOLD}✓ LÜFTER FUNKTIONIERT!{RESET}")
    else:
        print(f"  {RED}{BOLD}✗ LÜFTER HAT NICHT GESTARTET{RESET}")
        print(f"    → Schwelle (45°C) nicht erreicht oder Lüfter defekt")

    print(f"  Max. Temperatur: {max_temp:.1f}°C")
    print(f"  Aktuelle Temp:   {get_temp():.1f}°C")
    print()

if __name__ == "__main__":
    main()
