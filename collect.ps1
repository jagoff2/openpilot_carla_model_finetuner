$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$Host.UI.RawUI.WindowTitle='CARLA: concurrent dual-GPU dataset collection'
& '.\.venv\Scripts\python.exe' -u -m reports.automatic_corpus
if ($LASTEXITCODE -ne 0) {throw "Collector exited with code $LASTEXITCODE. Inspect reports/automatic/status.json under the corpus root."}
