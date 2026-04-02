param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$port = 8765
$baseUrl = "http://127.0.0.1:$port"
$healthUrl = "$baseUrl/api/health"
$appUrl = "$baseUrl/load/"
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$uiIndex = Join-Path $repoRoot "ui-web\out\index.html"

function Test-PeakService {
    param([string]$Url)
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
        return $response.Content -match '"status"\s*:\s*"ok"'
    } catch {
        return $false
    }
}

function Wait-ForPeakService {
    param(
        [string]$Url,
        [int]$TimeoutSeconds = 30
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-PeakService -Url $Url) {
            return $true
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Get-PortOwner {
    param([int]$LocalPort)
    try {
        return Get-NetTCPConnection -LocalPort $LocalPort -ErrorAction Stop |
            Select-Object -First 1 -ExpandProperty OwningProcess
    } catch {
        return $null
    }
}

if (-not (Test-Path $venvPython)) {
    Write-Host "Project virtual environment not found. Creating .venv..." -ForegroundColor Yellow
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -m venv .venv
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv .venv
    } else {
        throw "Python launcher not found. Install Python 3.11+ and retry."
    }
}

if (-not (Test-Path $uiIndex)) {
    Write-Host "Static UI build missing at ui-web\out. Build it with 'cd ui-web; npm install; npm run build'." -ForegroundColor Yellow
}

$portOwner = Get-PortOwner -LocalPort $port
if ($portOwner) {
    if (Test-PeakService -Url $healthUrl) {
        Write-Host "Peak Analysis Tool is already running on port $port." -ForegroundColor Green
        if (-not $NoBrowser) {
            Start-Process $appUrl | Out-Null
        }
        exit 0
    }

    $owner = Get-CimInstance Win32_Process -Filter "ProcessId = $portOwner" |
        Select-Object -ExpandProperty CommandLine
    throw "Port $port is already in use by another process: $owner"
}

$pythonExe = (Resolve-Path $venvPython).Path
Write-Host "Starting Peak Analysis Tool with $pythonExe" -ForegroundColor Cyan

$process = Start-Process -FilePath $pythonExe `
    -ArgumentList "-m", "service.app" `
    -WorkingDirectory $repoRoot `
    -PassThru

if (-not (Wait-ForPeakService -Url $healthUrl -TimeoutSeconds 30)) {
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
    }
    throw "Peak Analysis Tool did not become healthy within 30 seconds."
}

Write-Host "Peak Analysis Tool is ready at $appUrl" -ForegroundColor Green
if (-not $NoBrowser) {
    Start-Process $appUrl | Out-Null
}
