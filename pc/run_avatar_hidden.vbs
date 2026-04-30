Dim q
q = Chr(34)
Set oShell = CreateObject("WScript.Shell")
oShell.CurrentDirectory = "C:\Users\49179\moloch_repo\pc"
oShell.Run q & "C:\Users\49179\moloch_pc_env\Scripts\python.exe" & q & " " & q & "C:\Users\49179\moloch_repo\pc\avatar.py" & q, 0, False
