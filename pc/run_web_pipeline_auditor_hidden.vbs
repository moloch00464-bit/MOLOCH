' MOLOCH Web-Pipeline-Auditor (W19+W20a) — silent Daemon launcher (5min Loop)
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = sh.ExpandEnvironmentStrings("%USERPROFILE%\moloch_repo")
sh.Run "cmd /c """ & sh.ExpandEnvironmentStrings("%USERPROFILE%\moloch_pc_env\Scripts\python.exe") & """ -m pc.web_pipeline_auditor", 0, False
