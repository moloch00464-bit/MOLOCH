#!/usr/bin/env python3
"""
M.O.L.O.C.H. Hardware Dashboard

Live terminal dashboard showing:
- NPU Temperature & TOPS load
- NVMe write speed
- CPU core temperatures
- Audio channel status

Dark tech aesthetic using rich library.
M.O.L.O.C.H. knows its physical boundaries.
"""

import sys
import time
import signal
from pathlib import Path
from datetime import datetime

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.style import Style
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich import box

import status as telemetry

# Color scheme - dark tech aesthetic
COLORS = {
    "background": "#1a1a2e",
    "primary": "#e94560",
    "secondary": "#16213e",
    "accent": "#0f3460",
    "text": "#eaeaea",
    "ok": "#00ff88",
    "warning": "#ffaa00",
    "critical": "#ff4444",
    "offline": "#666666"
}


def get_status_color(status: str) -> str:
    """Get color for status level."""
    status_colors = {
        "ok": COLORS["ok"],
        "warning": COLORS["warning"],
        "critical": COLORS["critical"],
        "shutdown": COLORS["critical"],
        "offline": COLORS["offline"],
        "unknown": COLORS["offline"]
    }
    return status_colors.get(status, COLORS["text"])


def create_header() -> Panel:
    """Create the dashboard header."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header_text = Text()
    header_text.append("◉ ", style=f"bold {COLORS['primary']}")
    header_text.append("M.O.L.O.C.H. ", style=f"bold {COLORS['text']}")
    header_text.append("HARDWARE CONTROL CENTER", style=f"bold {COLORS['primary']}")
    header_text.append(f"\n{now}", style=f"dim {COLORS['text']}")

    return Panel(
        header_text,
        box=box.DOUBLE,
        style=f"bold {COLORS['primary']}",
        padding=(0, 2)
    )


def create_cpu_panel(data: dict) -> Panel:
    """Create CPU status panel."""
    cpu = data.get("cpu", {})

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Label", style=f"dim {COLORS['text']}")
    table.add_column("Value", style=f"bold {COLORS['text']}")

    # Temperature with color
    temp = cpu.get("temp_c", 0)
    temp_color = get_status_color(cpu.get("status", "unknown"))
    table.add_row("Temperature", Text(f"{temp}°C", style=f"bold {temp_color}"))

    # Usage
    usage = cpu.get("usage_percent", 0)
    usage_bar = create_usage_bar(usage)
    table.add_row("Usage", usage_bar)

    # Frequency
    freq = cpu.get("freq_mhz", 0)
    table.add_row("Frequency", f"{freq} MHz")

    # Per-core usage
    cores = cpu.get("per_core_usage", [])
    if cores:
        core_text = Text()
        for i, c in enumerate(cores):
            color = COLORS["ok"] if c < 80 else COLORS["warning"] if c < 95 else COLORS["critical"]
            core_text.append(f"C{i}:", style=f"dim {COLORS['text']}")
            core_text.append(f"{c:4.0f}% ", style=f"{color}")
        table.add_row("Cores", core_text)

    status_color = get_status_color(cpu.get("status", "unknown"))

    return Panel(
        table,
        title=f"[bold {COLORS['primary']}]◉ CPU (Pi 5)[/]",
        subtitle=f"[{status_color}]{cpu.get('status', 'unknown').upper()}[/]",
        box=box.ROUNDED,
        border_style=COLORS["accent"],
        padding=(1, 2)
    )


def create_npu_panel(data: dict) -> Panel:
    """Create NPU (Hailo) status panel."""
    npu = data.get("npu", {})

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Label", style=f"dim {COLORS['text']}")
    table.add_column("Value", style=f"bold {COLORS['text']}")

    # Online status
    online = npu.get("online", False)
    online_text = Text("● ONLINE" if online else "○ OFFLINE",
                       style=f"bold {COLORS['ok'] if online else COLORS['offline']}")
    table.add_row("Status", online_text)

    if online:
        # Board Temperature (RP1 chip near NPU)
        temp = npu.get("temp_c", 0)
        temp_color = get_status_color(npu.get("status", "unknown"))
        if temp > 0:
            table.add_row("Board Temp", Text(f"{temp:.1f}°C", style=f"bold {temp_color}"))

        # TOPS
        tops = npu.get("tops_available", 13)
        table.add_row("TOPS", f"{tops} TOPS")

    status_color = get_status_color(npu.get("status", "unknown"))

    return Panel(
        table,
        title=f"[bold {COLORS['primary']}]◉ NPU (Hailo-10H)[/]",
        subtitle=f"[{status_color}]{npu.get('status', 'unknown').upper()}[/]",
        box=box.ROUNDED,
        border_style=COLORS["accent"],
        padding=(1, 2)
    )


def create_storage_panel(data: dict) -> Panel:
    """Create storage status panel for multiple SSDs."""
    storage = data.get("storage", {})
    devices = storage.get("devices", [])

    # Fallback for old format (single device)
    if not devices and "used_gb" in storage:
        devices = [storage]

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Label", style=f"dim {COLORS['text']}")
    table.add_column("Value", style=f"bold {COLORS['text']}")

    for i, dev in enumerate(devices):
        if i > 0:
            table.add_row("", "")  # Spacer

        name = dev.get("name", f"Disk {i+1}")
        used = dev.get("used_gb", 0)
        total = dev.get("total_gb", 0)
        percent = dev.get("used_percent", 0)

        # Name header
        table.add_row(Text(f"▸ {name}", style=f"bold {COLORS['primary']}"), "")

        # Usage
        table.add_row("  Used", f"{used:.1f} / {total:.1f} GB")
        table.add_row("  Capacity", create_usage_bar(percent))

        # I/O speeds
        read_mbps = dev.get("read_mbps", 0)
        write_mbps = dev.get("write_mbps", 0)

        io_text = Text()
        io_text.append("R:", style=f"dim {COLORS['text']}")
        io_text.append(f"{read_mbps:5.1f} ", style=f"{COLORS['ok']}")
        io_text.append("W:", style=f"dim {COLORS['text']}")
        io_text.append(f"{write_mbps:5.1f} MB/s", style=f"{COLORS['primary']}")
        table.add_row("  I/O", io_text)

    status_color = get_status_color(storage.get("status", "unknown"))

    return Panel(
        table,
        title=f"[bold {COLORS['primary']}]◉ Storage[/]",
        subtitle=f"[{status_color}]{storage.get('status', 'unknown').upper()}[/]",
        box=box.ROUNDED,
        border_style=COLORS["accent"],
        padding=(1, 2)
    )


def create_ram_panel(data: dict) -> Panel:
    """Create RAM status panel."""
    ram = data.get("ram", {})

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Label", style=f"dim {COLORS['text']}")
    table.add_column("Value", style=f"bold {COLORS['text']}")

    # Usage
    used = ram.get("used_gb", 0)
    total = ram.get("total_gb", 0)
    percent = ram.get("used_percent", 0)
    available = ram.get("available_gb", 0)

    table.add_row("Used", f"{used:.2f} / {total:.2f} GB")
    table.add_row("Available", f"{available:.2f} GB")
    table.add_row("Usage", create_usage_bar(percent))

    status_color = get_status_color(ram.get("status", "unknown"))

    return Panel(
        table,
        title=f"[bold {COLORS['primary']}]◉ RAM[/]",
        subtitle=f"[{status_color}]{ram.get('status', 'unknown').upper()}[/]",
        box=box.ROUNDED,
        border_style=COLORS["accent"],
        padding=(1, 2)
    )


def create_audio_panel(data: dict) -> Panel:
    """Create audio channels status panel."""
    audio = data.get("audio", {})
    channels = audio.get("channels", [])

    table = Table(show_header=True, box=box.SIMPLE, padding=(0, 1))
    table.add_column("CH", style=f"dim {COLORS['text']}", width=3)
    table.add_column("Name", style=f"{COLORS['text']}", width=12)
    table.add_column("Status", width=8)
    table.add_column("Vol", width=4)

    for i, ch in enumerate(channels[:8]):
        ch_id = ch.get("id", i)
        name = ch.get("name", f"CH{i}")[:12]
        status = ch.get("status", "offline")
        volume = ch.get("volume_percent", 0)

        status_color = get_status_color(status)
        status_icon = "●" if status == "ok" else "○"

        table.add_row(
            f"{ch_id}",
            name,
            Text(f"{status_icon} {status[:6]}", style=status_color),
            f"{volume}%" if status == "ok" else "-"
        )

    active = audio.get("active_count", 0)
    status_color = get_status_color(audio.get("status", "unknown"))

    return Panel(
        table,
        title=f"[bold {COLORS['primary']}]◉ Audio Channels ({active}/8 active)[/]",
        subtitle=f"[{status_color}]{audio.get('status', 'unknown').upper()}[/]",
        box=box.ROUNDED,
        border_style=COLORS["accent"],
        padding=(1, 2)
    )


def create_fan_panel(data: dict) -> Panel:
    """Create fan status panel."""
    fan = data.get("fan", {})

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Label", style=f"dim {COLORS['text']}")
    table.add_column("Value", style=f"bold {COLORS['text']}")

    state = fan.get("state", 0)
    max_state = fan.get("max_state", 4)
    speed_pct = fan.get("speed_percent", 0)
    temp = fan.get("temp_c", 0)
    status = fan.get("status", "unknown")

    # Fan speed indicator
    if state == 0:
        speed_text = Text("○ AUS", style=f"dim {COLORS['offline']}")
    elif state == 1:
        speed_text = Text("◐ NIEDRIG", style=f"{COLORS['ok']}")
    elif state == 2:
        speed_text = Text("◑ MITTEL", style=f"{COLORS['ok']}")
    elif state == 3:
        speed_text = Text("● HOCH", style=f"{COLORS['warning']}")
    else:
        speed_text = Text("● MAX", style=f"{COLORS['critical']}")

    table.add_row("Speed", speed_text)

    # PWM State as visual bar
    fan_bar = Text()
    for i in range(max_state):
        if i < state:
            fan_bar.append("▮", style=f"{COLORS['primary']}")
        else:
            fan_bar.append("▯", style=f"dim {COLORS['offline']}")
    fan_bar.append(f" {state}/{max_state}", style=f"dim {COLORS['text']}")
    table.add_row("PWM", fan_bar)

    # Temperature
    temp_color = get_status_color(status)
    table.add_row("CPU Temp", Text(f"{temp:.1f}°C", style=f"bold {temp_color}"))

    status_color = get_status_color(status)

    return Panel(
        table,
        title=f"[bold {COLORS['primary']}]◉ Fan[/]",
        subtitle=f"[{status_color}]{status.upper()}[/]",
        box=box.ROUNDED,
        border_style=COLORS["accent"],
        padding=(1, 2)
    )


def create_usage_bar(percent: float, width: int = 20) -> Text:
    """Create a colored usage bar."""
    filled = int((percent / 100) * width)
    empty = width - filled

    if percent < 60:
        color = COLORS["ok"]
    elif percent < 80:
        color = COLORS["warning"]
    else:
        color = COLORS["critical"]

    bar = Text()
    bar.append("█" * filled, style=color)
    bar.append("░" * empty, style=f"dim {COLORS['offline']}")
    bar.append(f" {percent:5.1f}%", style=f"bold {color}")

    return bar


def create_safety_panel(data: dict) -> Panel:
    """Create safety status panel."""
    overall = data.get("overall_status", "unknown")
    status_color = get_status_color(overall)

    # Safety check
    engine = telemetry.get_engine()
    is_safe, reason = engine.check_safety()

    if is_safe:
        icon = "✓"
        message = "All systems within safe operating limits"
    else:
        icon = "⚠" if "CRITICAL" in reason else "✕"
        message = reason

    content = Text()
    content.append(f" {icon} ", style=f"bold {status_color}")
    content.append(message, style=f"{COLORS['text']}")

    return Panel(
        content,
        title=f"[bold {COLORS['primary']}]◉ Safety Status[/]",
        subtitle=f"[{status_color}]{overall.upper()}[/]",
        box=box.ROUNDED,
        border_style=status_color,
        padding=(0, 2)
    )


def create_dashboard(data: dict) -> Layout:
    """Create the complete dashboard layout."""
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="main"),
        Layout(name="footer", size=4)
    )

    layout["main"].split_row(
        Layout(name="left"),
        Layout(name="right")
    )

    layout["left"].split_column(
        Layout(name="cpu"),
        Layout(name="npu_fan")
    )

    layout["left"]["npu_fan"].split_row(
        Layout(name="npu"),
        Layout(name="fan")
    )

    layout["right"].split_column(
        Layout(name="storage"),
        Layout(name="ram_audio")
    )

    layout["right"]["ram_audio"].split_row(
        Layout(name="ram"),
        Layout(name="audio")
    )

    # Populate layout
    layout["header"].update(create_header())
    layout["cpu"].update(create_cpu_panel(data))
    layout["npu"].update(create_npu_panel(data))
    layout["fan"].update(create_fan_panel(data))
    layout["storage"].update(create_storage_panel(data))
    layout["ram"].update(create_ram_panel(data))
    layout["audio"].update(create_audio_panel(data))
    layout["footer"].update(create_safety_panel(data))

    return layout


def run_dashboard(refresh_rate: float = 1.0):
    """Run the live dashboard."""
    console = Console()

    # Handle Ctrl+C gracefully
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Clear screen
    console.clear()

    engine = telemetry.get_engine()

    try:
        with Live(console=console, refresh_per_second=1/refresh_rate, screen=True) as live:
            while running:
                try:
                    data = engine.to_dict()
                    dashboard = create_dashboard(data)
                    live.update(dashboard)

                    # Safety check - log warnings
                    is_safe, reason = engine.check_safety()
                    if not is_safe:
                        from core import status
                        status.logger.warning(reason)

                    time.sleep(refresh_rate)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                    time.sleep(1)

    except Exception as e:
        console.print(f"[red]Dashboard error: {e}[/red]")

    console.clear()
    console.print("[bold]M.O.L.O.C.H. Dashboard closed.[/bold]")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="M.O.L.O.C.H. Hardware Dashboard")
    parser.add_argument("-r", "--refresh", type=float, default=1.0,
                        help="Refresh rate in seconds (default: 1.0)")
    parser.add_argument("--once", action="store_true",
                        help="Show dashboard once and exit")

    args = parser.parse_args()

    if args.once:
        console = Console()
        engine = telemetry.get_engine()
        data = engine.to_dict()
        dashboard = create_dashboard(data)
        console.print(dashboard)
    else:
        run_dashboard(args.refresh)


if __name__ == "__main__":
    main()
