# Setzt Windows OS-Permission "Microphone" fuer Chrome auf Allow.
#
# Hintergrund: Win10/11 hat einen separaten Privacy-Layer pro Desktop-App.
# Wenn dieser leer ist, blockiert das OS Mic-Zugriff egal was Chrome's
# eigene Site-Settings sagen — und die Site-UI ist dann ausgegraut.
#
# HKCU = kein Admin noetig.
# WICHTIG: Chrome muss komplett neu gestartet werden (alle Tabs, alle
# Hintergrund-Prozesse), damit die Permission greift.

$key = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone\NonPackaged\C:#Program Files#Google#Chrome#Application#chrome.exe'

if (-not (Test-Path $key)) {
    Write-Output "[mic] Chrome consent key not found:"
    Write-Output "      $key"
    Write-Output "[mic] Chrome wurde wahrscheinlich noch nie gestartet (oder nicht im Standard-Pfad). Abbruch."
    exit 1
}

$before = (Get-ItemProperty -Path $key -ErrorAction SilentlyContinue).Value
Set-ItemProperty -Path $key -Name 'Value' -Value 'Allow' -Type String -Force

$ts = [DateTime]::UtcNow.ToFileTimeUtc()
Set-ItemProperty -Path $key -Name 'LastUsedTimeStart' -Value $ts -Type QWord -Force -ErrorAction SilentlyContinue
Set-ItemProperty -Path $key -Name 'LastUsedTimeStop'  -Value $ts -Type QWord -Force -ErrorAction SilentlyContinue

Write-Output "[mic] Chrome microphone consent: '$before' -> 'Allow'"
Write-Output ""
Write-Output "WICHTIG: Chrome komplett schliessen (alle Fenster + Hintergrund-Prozesse,"
Write-Output "         z.B. Taskleiste -> Chrome -> 'Beenden'), dann neu oeffnen."
Write-Output "         Erst danach greift die neue Permission."
