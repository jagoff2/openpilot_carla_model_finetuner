[CmdletBinding()]
param([ValidateSet('all','doctor','tests','calibrate','plan')][string]$Stage='all')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
& '.\.venv\Scripts\python.exe' -u -m pipeline.bootstrap --stage $Stage
if ($LASTEXITCODE -ne 0) {throw "Preparation stage $Stage failed. Inspect the reported error before collection."}
