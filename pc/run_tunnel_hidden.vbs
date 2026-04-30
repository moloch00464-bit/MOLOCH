Dim q
q = Chr(34)
Set oShell = CreateObject("WScript.Shell")
oShell.CurrentDirectory = "C:\Users\49179\moloch_repo\pc"
oShell.Run "cmd /c " & q & "C:\Users\49179\moloch_repo\pc\start_pi_tunnel.bat" & q, 0, False
