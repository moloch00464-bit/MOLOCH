' MOLOCH Persona-Validator — silent launcher (Welle 10)
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = sh.ExpandEnvironmentStrings("%USERPROFILE%\moloch_repo\pc")
sh.Run "cmd /c """ & sh.ExpandEnvironmentStrings("%USERPROFILE%\moloch_pc_env\Scripts\python.exe") & """ persona_validator.py --interval-s 10", 0, False
