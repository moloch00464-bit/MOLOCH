' MOLOCH Mailbox-Auditor — silent launcher (no terminal popup)
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = sh.ExpandEnvironmentStrings("%USERPROFILE%\moloch_repo\pc")
sh.Run "cmd /c """ & sh.ExpandEnvironmentStrings("%USERPROFILE%\moloch_pc_env\Scripts\python.exe") & """ mailbox_auditor.py --interval-s 300", 0, False
