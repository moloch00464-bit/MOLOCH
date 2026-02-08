# M.O.L.O.C.H. Readiness Console

## Philosophy

The Readiness Console is the **interface**, not the **intelligence**.

In Phase 2, M.O.L.O.C.H. has no understanding. The console is a shell - a stable point of contact between human and system. It receives input, logs it, and provides basic system information. Nothing more.

**What the console IS:**
- A text-based interface
- A command handler for system queries
- An input logger
- A status display

**What the console is NOT:**
- A chatbot
- An AI assistant
- A semantic processor
- A decision maker

## Design Principles

### 1. Stability Over Features
The console must always work. No crashes, no hangs, no mysterious behavior.

### 2. Transparency
All input is logged. All output is visible. No hidden processing.

### 3. No Autonomy
The console does not act on its own. It waits. It responds to commands. It acknowledges input. That's all.

### 4. Foundation for Future
This console will eventually be the entry point for AI interaction. But the AI comes later. The console must be solid first.

## Usage

### Starting the Console

```bash
cd ~/moloch
python3 core/console/moloch_console.py
```

Or make it executable:
```bash
chmod +x ~/moloch/core/console/moloch_console.py
~/moloch/core/console/moloch_console.py
```

### Console Interface

```
======================================================================
              [ M.O.L.O.C.H. | Phase 2 | Idle ]
======================================================================

M.O.L.O.C.H. Readiness Console active.
Type /help for available commands.

>
```

### Built-in Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/status` | Show system status (CPU, RAM, temp) |
| `/world` | Show world inventory summary |
| `/inventory` | Show detailed inventory |
| `/history` | Show input history |
| `/clear` | Clear the screen |
| `/version` | Show version info |
| `/exit` | Exit the console |

### Non-Command Input

Any input that doesn't start with `/` is:
1. Logged to `~/moloch/logs/console.log`
2. Acknowledged with a "received" message
3. Not processed semantically

Example:
```
> Hello M.O.L.O.C.H.

[Received] "Hello M.O.L.O.C.H."
[Phase 2] Input logged. No semantic processing active.

>
```

## Logging

All console activity is logged to:
```
~/moloch/logs/console.log
```

Log format:
```
[2026-01-19 10:30:45,123] INFO: Console session started
[2026-01-19 10:30:50,456] INFO: Input: Hello M.O.L.O.C.H.
[2026-01-19 10:31:00,789] INFO: Input: /status
```

## Integration Points

### World Inventory
The console can display the world inventory via `/world` and `/inventory` commands. This shows what peripherals are available or planned.

### Future: AI Integration
In later phases, the console will:
- Pass non-command input to an AI processor
- Display AI responses
- Support conversation history
- Enable voice input/output toggle

But not yet. Phase 2 is readiness only.

## File Structure

```
~/moloch/core/console/
├── moloch_console.py    # Main console application
├── README.md            # This file
└── (future: themes/)    # Console themes
```

## Configuration

Current configuration (in `moloch_console.py`):

```python
CONSOLE_CONFIG = {
    "header": "[ M.O.L.O.C.H. | Phase 2 | Idle ]",
    "prompt": "> ",
    "width": 70,
    "phase": 2
}
```

## Security Considerations

- Console runs with user permissions only
- No network access
- No system modifications
- Input is logged but not executed
- Commands are whitelisted (no arbitrary execution)

## Troubleshooting

### Console won't start
```bash
# Check Python
python3 --version

# Check dependencies
python3 -c "import sys; print(sys.path)"

# Run with verbose errors
python3 -u ~/moloch/core/console/moloch_console.py
```

### Commands not working
```bash
# Check world inventory exists
cat ~/moloch/core/world/state/world_inventory.json

# Check logs for errors
tail ~/moloch/logs/console.log
```

### Screen messed up
```
/clear
```
Or press Ctrl+L in most terminals.

## The Waiting

> "M.O.L.O.C.H. wartet. Nicht weil es muss, sondern weil es noch nicht Zeit ist zu handeln. Die Konsole ist bereit. Der Mensch kann kommen."

---

**Phase 2: The Interface Exists. The Mind Comes Later.** 🖥️
