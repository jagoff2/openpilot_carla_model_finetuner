param(
    [ValidateSet('Prepare','Benchmark','Develop','Generate','Validate','Export','Status')]
    [string]$Mode = 'Status',
    [string]$DatasetRoot = '',
    [ValidateRange(1,2)][int]$Workers = 2,
    [ValidateRange(4,10000)][int]$Ticks = 500,
    [ValidateRange(0.1,30)][double]$Speed = 5,
    [ValidateSet(0,2,3,4)][int]$Traffic = 0
)
$ErrorActionPreference = 'Stop'
function Test-ExactCommandSetting([string]$CommandLine, [string]$Setting) {
    $parts = $Setting.Split('=', 2)
    $pattern = '(?:^|[\s",=])' + [regex]::Escape($parts[0]) + '=([^\s",]+)'
    $matches = [regex]::Matches($CommandLine, $pattern, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
    return $matches.Count -eq 1 -and $matches[0].Groups[1].Value -eq $parts[1]
}
$projectPath = $PSScriptRoot
$pythonPath = Join-Path $projectPath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project Python environment is missing. Run setup.ps1.' }
$settings = & $pythonPath -c 'import json; from lane_dataset.config import Config,DEFAULT_ROOT; print(json.dumps(dict(carla_root=Config().carla_root,dataset_root=str(DEFAULT_ROOT))))' | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) {throw 'Could not load local settings.'}
if (-not $DatasetRoot) {$DatasetRoot=$settings.dataset_root}
$serverPath=Join-Path $settings.carla_root 'CarlaUE4.exe'
Push-Location $projectPath
try {
    if ($Mode -in @('Prepare','Validate','Export','Status')) {
        $command = $Mode.ToLowerInvariant()
        & $pythonPath -m lane_dataset --root $DatasetRoot $command
        if ($LASTEXITCODE -ne 0) { throw "$command failed with exit code $LASTEXITCODE" }
        return
    }
    & $pythonPath -m lane_dataset --root $DatasetRoot recover
    if ($LASTEXITCODE -ne 0) { throw 'Recovery failed' }
    $sessionDirectory = Join-Path $DatasetRoot ('reports\launcher-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ'))
    New-Item -ItemType Directory -Path $sessionDirectory -Force | Out-Null
    $launchRows = @()
    for ($workerIndex = 0; $workerIndex -lt $Workers; $workerIndex++) {
        $rpcPort = 2400 + 100 * $workerIndex
        $portOwner = Get-NetTCPConnection -State Listen -LocalPort $rpcPort -ErrorAction SilentlyContinue
        if ($portOwner) {
            $processRecord = Get-CimInstance Win32_Process -Filter "ProcessId=$($portOwner[0].OwningProcess)"
            if ($processRecord.Name -ne 'CarlaUE4-Win64-Shipping.exe' -or -not (Test-ExactCommandSetting $processRecord.CommandLine "-carla-rpc-port=$rpcPort")) {
                throw "Port $rpcPort belongs to an unrelated process"
            }
            foreach ($requiredArgument in @("-graphicsadapter=$workerIndex", '-quality-level=Epic', 'r.DynamicRes.OperationMode=0', 'r.ScreenPercentage=100')) {
                if (-not (Test-ExactCommandSetting $processRecord.CommandLine $requiredArgument)) {
                    throw "Existing CARLA on port $rpcPort does not declare required setting $requiredArgument"
                }
            }
            $serverProcessId = $processRecord.ProcessId
        } else {
            $serverArguments = @('-RenderOffScreen','-nosound','-quality-level=Epic',"-graphicsadapter=$workerIndex", "-carla-rpc-port=$rpcPort",'-NoTextureStreaming','-ExecCmds=r.DynamicRes.OperationMode=0,r.ScreenPercentage=100')
            $serverProcess = Start-Process -FilePath $serverPath -ArgumentList $serverArguments -WindowStyle Hidden -PassThru
            $serverProcessId = $serverProcess.Id
        }
        $launchRows += @{ worker=$workerIndex; gpu=$workerIndex; rpc_port=$rpcPort; server_pid=$serverProcessId }
    }
    $launchRows | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $sessionDirectory 'workers.json') -Encoding UTF8
    & $pythonPath -m lane_dataset.benchmark --root $DatasetRoot --workers $Workers --ticks $Ticks --speed $Speed --traffic $Traffic --mode $Mode.ToLowerInvariant() --report-directory $sessionDirectory
    if ($LASTEXITCODE -ne 0) { throw "$Mode failed; inspect $sessionDirectory" }
} finally {
    Pop-Location
}
