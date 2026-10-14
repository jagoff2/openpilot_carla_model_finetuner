$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$Host.UI.RawUI.WindowTitle='CD210: centerline fine-tuning while data arrives'
& '.\.venv-training\Scripts\python.exe' -u -m lane_training.production
if ($LASTEXITCODE -ne 0) {throw "Trainer exited with code $LASTEXITCODE. Inspect runs/cd210-centerline-production/status.json."}
