# Build moloch-coder Modell aus Modelfile
$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $here

Write-Host "[build] ollama create moloch-coder ..."
ollama create moloch-coder -f Modelfile

Write-Host ""
Write-Host "[build] Skill-Match-Test ..."
python prompt_builder.test.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[build] FAIL: skill-match"
    exit 1
}

Write-Host ""
Write-Host "[build] Smoketest 1: Identitaet ..."
$resp1 = ollama run moloch-coder "Wer bist du? Antworte in maximal 2 Saetzen." 2>&1
Write-Host $resp1

Write-Host ""
Write-Host "[build] Smoketest 2: Audit-Pattern via prompt_builder ..."
python prompt_builder.py "schreib einen example_auditor stub mit collect()"

Write-Host ""
Write-Host "[build] DONE."
