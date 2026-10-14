[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$CarlaRoot,
    [string]$DatasetRoot=(Join-Path $PSScriptRoot 'data\corpus'),
    [string]$CapturePython='',
    [string]$TrainingPython='',
    [string]$SourceExports='',
    [string]$SourceBundle=''
)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
function Invoke-Checked([string]$Executable,[string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {throw "Command failed ($LASTEXITCODE): $Executable $Arguments"}
}
$CarlaRoot=(Resolve-Path -LiteralPath $CarlaRoot).Path
if (-not (Test-Path -LiteralPath (Join-Path $CarlaRoot 'CarlaUE4.exe'))) {throw 'CARLA 0.9.16 Windows installation must contain CarlaUE4.exe.'}
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if ($CapturePython) {Invoke-Checked $CapturePython @('-m','venv','.venv')}
    else {Invoke-Checked 'py' @('-3.12','-m','venv','.venv')}
}
if (-not (Test-Path -LiteralPath '.venv-training\Scripts\python.exe')) {
    if ($TrainingPython) {Invoke-Checked $TrainingPython @('-m','venv','.venv-training')}
    else {Invoke-Checked 'py' @('-3.10','-m','venv','.venv-training')}
}
$capture=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$training=Join-Path $PSScriptRoot '.venv-training\Scripts\python.exe'
Invoke-Checked $capture @('-m','pip','install','--upgrade','pip')
Invoke-Checked $training @('-m','pip','install','--upgrade','pip')
Invoke-Checked $capture @('-m','pip','install','-e','.[test]')
$carlaWheels=@(Get-ChildItem -LiteralPath (Join-Path $CarlaRoot 'PythonAPI\carla\dist') -Filter '*0.9.16*cp312*win_amd64.whl' -ErrorAction SilentlyContinue)
if ($carlaWheels.Count -eq 1) {Invoke-Checked $capture @('-m','pip','install',$carlaWheels[0].FullName)}
else {Invoke-Checked $capture @('-m','pip','install','carla==0.9.16')}
# CARLA owns both GPUs. Install the CPU training runtime deliberately.
Invoke-Checked $training @('-m','pip','install','torch>=2.6,<3','--index-url','https://download.pytorch.org/whl/cpu')
Invoke-Checked $training @('-m','pip','install','-e','.[training,test]')
$settings=@{carla_root=$CarlaRoot;dataset_root=[IO.Path]::GetFullPath($DatasetRoot);capture_python=$capture;training_python=$training}
$settings | ConvertTo-Json | Set-Content -LiteralPath 'config.local.json' -Encoding UTF8
$fetchArguments=@('-u','-m','lane_training.fetch_cd210')
if ($SourceExports) {$fetchArguments+=@('--source-exports',$SourceExports)}
if ($SourceBundle) {$fetchArguments+=@('--source-bundle',$SourceBundle)}
Invoke-Checked $training $fetchArguments
Write-Host 'Setup complete. Run .\prepare.ps1 before starting collection on this corpus.'
