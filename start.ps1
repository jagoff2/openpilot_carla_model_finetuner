[CmdletBinding()]
param([switch]$Prepare)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
if ($Prepare) {& '.\prepare.ps1'}
foreach ($scriptName in @('collect.ps1','train.ps1')) {
    $scriptPath=Join-Path $PSScriptRoot $scriptName
    Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-NoExit','-ExecutionPolicy','Bypass','-File',('"'+$scriptPath+'"')) -WorkingDirectory $PSScriptRoot -WindowStyle Normal
}
Write-Host 'Collector and trainer launched concurrently in separate consoles. Check their status files for progress.'
