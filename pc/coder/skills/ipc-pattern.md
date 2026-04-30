# IPC-Pattern

## Action registrieren

```python
# moloch_service.py
register_action("spotify_action_naechster_song",
                handler=lambda params: spotify.next_track())
```

Naming-Konvention: `<domain>_<verb>` (spotify_play, ptz_pan, led_set).

## Action triggern

```python
# ipc_router.py
result = route_action("spotify_action_naechster_song", params={})
```

Params via JSON-Dict oder stdin.

## NEVER 5: subprocess timeout

```python
subprocess.run(cmd, timeout=30, capture_output=True)
```

NIEMALS ohne timeout — sonst hangt der Service.

## NEVER 8: kein shell=True

```python
# RICHTIG
subprocess.run(["python", "script.py", arg], timeout=30)

# FALSCH
subprocess.run("python script.py " + arg, shell=True)  # injection!
```

## Klassifikator-Pattern (Bug B)

`chat_server._classify_prompt_type` entscheidet `prompt_type` VOR Routing.
Action-Shortcuts (z.B. "naechster Song") muessen VOR LLM-Routing matchen,
sonst landen sie als `music_query` beim Tentakel statt als IPC-Trigger.

Lage: `core/chat/chat_server.py:_classify_prompt_type`
