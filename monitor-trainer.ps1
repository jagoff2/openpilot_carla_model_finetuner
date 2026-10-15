param(
    [string]$RunDirectory = (Join-Path $PSScriptRoot 'runs\cd210-centerline-production'),
    [switch]$Once
)

$Host.UI.RawUI.WindowTitle = 'CD210 trainer - live status'
$statusPath = Join-Path $RunDirectory 'status.json'
$previous = ''
$lastPrinted = [datetime]::MinValue
Write-Host 'CD210 trainer status monitor. Closing this window leaves training running.' -ForegroundColor Cyan
Write-Host ('Run: ' + $RunDirectory)
Write-Host 'Training resumes automatically when new accepted training episodes arrive.'
Write-Host 'Validation and test episodes are excluded from optimizer updates.'

do {
    try {
        $status = Get-Content -LiteralPath $statusPath -Raw -ErrorAction Stop | ConvertFrom-Json
        $trainerProcess = Get-Process -Id ([int]$status.pid) -ErrorAction SilentlyContinue
        $age = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - [double]$status.updated
        $processState = if ($null -ne $trainerProcess) { 'alive' } else { 'EXITED' }
        $summary = 'PID {0} {1} | {2} | optimizer steps {3} | trained episodes {4} | device {5}' -f $status.pid, $processState, $status.phase, $status.optimizer_steps, $status.trained_episodes, $status.device
        $active = if ($null -ne $status.active) { $status.active | ConvertTo-Json -Compress -Depth 6 } else { '' }
        $signature = $summary + $active
        if ($signature -ne $previous -or ((Get-Date) - $lastPrinted).TotalSeconds -ge 20) {
            $color = if ($null -eq $trainerProcess -or $age -gt 60) { 'Yellow' } else { 'Green' }
            Write-Host ('[{0}] {1} | status age {2:N0}s' -f (Get-Date -Format 'HH:mm:ss'), $summary, $age) -ForegroundColor $color
            if ($active) { Write-Host ('  Active: ' + $active) }
            $collectionPath = Join-Path $status.training_config.corpus 'reports\automatic\status.json'
            try {
                $collection = Get-Content -LiteralPath $collectionPath -Raw -ErrorAction Stop | ConvertFrom-Json
                Write-Host ('  Accepted labels: train {0}, validation {1}, test {2}; total {3}/5000' -f $collection.accepted_labels.train, $collection.accepted_labels.validation, $collection.accepted_labels.test, $collection.accepted_total)
            } catch { Write-Host ('  Collection status temporarily unavailable: ' + $_.Exception.Message) -ForegroundColor Yellow }
            if ($null -ne $status.last_validation) {
                Write-Host ('  Last held-out lateral loss: candidate {0:N6}, stock {1:N6}, step {2}' -f $status.last_validation.loss, $status.last_validation.baseline_loss, $status.last_validation.step)
            }
            $previous = $signature
            $lastPrinted = Get-Date
        }
    } catch {
        Write-Host ('[{0}] Cannot read trainer status: {1}' -f (Get-Date -Format 'HH:mm:ss'), $_.Exception.Message) -ForegroundColor Yellow
    }
    if (-not $Once) { Start-Sleep -Seconds 2 }
} while (-not $Once)
