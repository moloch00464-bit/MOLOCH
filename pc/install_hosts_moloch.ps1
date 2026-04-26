# Setzt hosts-Eintrag fuer moloch.local -> 192.168.178.30
# Run as Admin: rechts-klick -> "Mit PowerShell ausfuehren" oder via
# Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File install_hosts_moloch.ps1"

$entry = "192.168.178.30 moloch.local"
$hostsFile = "$env:windir\System32\drivers\etc\hosts"

$existing = Select-String -Path $hostsFile -Pattern "moloch\.local" -ErrorAction SilentlyContinue
if ($existing) {
    Write-Output "[hosts] moloch.local already mapped:"
    Write-Output "        $($existing.Line)"
    return
}

try {
    Add-Content -Path $hostsFile -Value $entry -Encoding ASCII -ErrorAction Stop
    Write-Output "[hosts] added: $entry"
} catch {
    Write-Output ("[hosts] FAILED: " + $_.Exception.Message)
    Write-Output "[hosts] hint: as Administrator?"
    exit 1
}
