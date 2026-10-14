[CmdletBinding()]
param([switch]$OnnxOnly,[string]$DeviceHost='iq@192.168.1.95',[string]$Resume='')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$exportPython=Join-Path $PSScriptRoot '.venv-training\Scripts\python.exe'
$exportArguments=@('-u','-m','lane_release.export','--device',$DeviceHost)
if ($OnnxOnly) {$exportArguments+='--onnx-only'}
if ($Resume) {$exportArguments+=@('--resume',$Resume)}
& $exportPython @exportArguments
if ($LASTEXITCODE -eq 3) {Write-Host "ONNX export retained. c3x compilation is pending until $DeviceHost is online. Use -Resume with the printed release directory."; exit 3}
if ($LASTEXITCODE -ne 0) {throw 'Export incomplete. Verified ONNX and device diagnostics are retained in releases\cd210.'}
