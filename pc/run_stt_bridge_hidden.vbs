' MOLOCH STT-Bridge (faster-whisper) — silent Daemon launcher
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = sh.ExpandEnvironmentStrings("%USERPROFILE%\moloch_repo\pc")
sh.Run "cmd /c """ & sh.ExpandEnvironmentStrings("%USERPROFILE%\moloch_pc_env\Scripts\python.exe") & """ stt_bridge.py", 0, False
