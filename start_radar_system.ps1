# RADAR one-click startup script
# Starts Flask (5000) + OHIF viewer (3002) in the background and streams
# both services' logs into THIS terminal.
#   .\start_radar_system.ps1          start + tail logs
#   .\start_radar_system.ps1 -Stop    stop the services started by this script
param(
    [switch]$Stop
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$FlaskDir = Join-Path $ProjectRoot 'flask-app'
$OhifDir = Join-Path $ProjectRoot 'ohif-viewer\platform\app'
$LogDir = Join-Path $ProjectRoot 'logs'
$FlaskLog = Join-Path $LogDir 'flask.log'
$OhifLog = Join-Path $LogDir 'ohif.log'
$PidFile = Join-Path $LogDir 'services.pid'

# OHIF port. Ports 3000/3001 are taken by local Docker containers
# (3000 = apisix-prod-grafana-1, 3001 = bzh-server-bzh-server-1). Reusing an
# occupied port makes the Test-Port check below wrongly assume "OHIF is already
# running" and skip startup, so the frontend iframe ends up hitting the other
# service and the viewer throws "Invalid study URL".
# Must match OHIF_PORT in flask-app\app.py and static\js\app.js.
$OhifPort = 3002

function Test-Port($Port) {
    try {
        $connection = Test-NetConnection -ComputerName '127.0.0.1' -Port $Port -WarningAction SilentlyContinue
        return $connection.TcpTestSucceeded
    } catch {
        return $false
    }
}

function Get-PortOwner($Port) {
    try {
        $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($conn) { return $conn.OwningProcess }
    } catch { }
    return $null
}

function New-EmptyLog($Path) {
    # Create/truncate WITHOUT a BOM. Set-Content -Encoding utf8 writes a BOM on
    # PowerShell 5, which shows up as a stray blank line when tailing the log.
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, '', $utf8NoBom)
}

# Tail several log files at once and print each new line with a coloured prefix.
# Get-Content -Wait does not reliably follow more than one file, so poll manually.
function Watch-Logs {
    param(
        [string[]]$Paths,
        [string[]]$Prefixes,
        [string[]]$Colors
    )
    $offsets = @{}
    foreach ($path in $Paths) { $offsets[$path] = 0 }

    while ($true) {
        for ($i = 0; $i -lt $Paths.Count; $i++) {
            $path = $Paths[$i]
            if (-not (Test-Path $path)) { continue }

            $stream = $null
            try {
                $stream = [System.IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
                $len = $stream.Length
                $pos = $offsets[$path]
                if ($len -lt $pos) { $pos = 0 }          # log was truncated / restarted
                if ($len -gt $pos) {
                    $stream.Seek($pos, 'Begin') | Out-Null
                    $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8)
                    $stream = $null                      # reader now owns the stream
                    while (($line = $reader.ReadLine()) -ne $null) {
                        if ($line.Trim().Length -eq 0) { continue }
                        Write-Host ('[' + $Prefixes[$i] + '] ') -NoNewline -ForegroundColor $Colors[$i]
                        Write-Host $line
                    }
                    $offsets[$path] = $reader.BaseStream.Position
                    $reader.Dispose()
                } else {
                    $stream.Dispose()
                }
            } catch {
                if ($stream) { $stream.Dispose() }
            }
        }
        Start-Sleep -Milliseconds 500
    }
}

# ---------------------------------------------------------------- stop mode
if ($Stop) {
    $targets = @()
    if (Test-Path $PidFile) {
        $targets += Get-Content $PidFile | Where-Object { $_ -match '^\d+$' }
    }
    foreach ($port in 5000, $OhifPort) {
        $owner = Get-PortOwner $port
        if ($owner) { $targets += $owner }
    }
    $targets = $targets | Select-Object -Unique
    if (-not $targets) {
        Write-Host '[INFO] No running RADAR service found.' -ForegroundColor Yellow
        return
    }
    foreach ($procId in $targets) {
        $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if (-not $proc) { continue }
        if ($proc.ProcessName -notin @('python', 'node', 'powershell', 'conda')) {
            Write-Host "[SKIP] pid $procId is '$($proc.ProcessName)', not a RADAR service." -ForegroundColor Yellow
            continue
        }
        Write-Host "[STOP] pid $procId ($($proc.ProcessName))" -ForegroundColor Yellow
        # /T also kills child processes (npm -> node, conda -> python)
        & taskkill.exe /PID $procId /T /F 2>&1 | Out-Null
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    Write-Host 'Services stopped.' -ForegroundColor Green
    return
}

# --------------------------------------------------------------- start mode
Write-Host '============================================' -ForegroundColor Cyan
Write-Host ' RADAR Medical Imaging Platform - Startup' -ForegroundColor Cyan
Write-Host '============================================' -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"
Write-Host "Logs:    $LogDir"
Write-Host ''

if (-not (Test-Path (Join-Path $FlaskDir 'app.py'))) {
    throw "Flask app not found: $FlaskDir\app.py"
}

if (-not (Test-Path (Join-Path $OhifDir 'package.json'))) {
    throw "OHIF app not found: $OhifDir\package.json"
}

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw 'conda was not found. Please add Anaconda or Miniconda to PATH.'
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw 'npm was not found. Please install Node.js and add it to PATH.'
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$startedPids = @()

if (Test-Port 5000) {
    Write-Host '[INFO] Flask port 5000 is already in use. Reusing the existing service.' -ForegroundColor Yellow
    Write-Host '       (its logs stay in the terminal that started it)' -ForegroundColor DarkGray
} else {
    Write-Host '[1/2] Starting Flask backend...' -ForegroundColor Green
    New-EmptyLog $FlaskLog
    # PYTHONUNBUFFERED=1 -> Python flushes stdout immediately so the tail below is real-time.
    # Use Out-File directly: the Watch-Logs tail adds the [flask] prefix.
    # NOTE: keep this command single-quote only, Start-Process mangles embedded double quotes.
    $flaskCmd = "`$env:PYTHONUNBUFFERED='1'; conda run --no-capture-output -n radar python '$FlaskDir\app.py' 2>&1 | " +
        "Out-File -FilePath '$FlaskLog' -Append -Encoding utf8"
    $proc = Start-Process powershell.exe -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $flaskCmd
    )
    $startedPids += $proc.Id
}

if (Test-Port $OhifPort) {
    Write-Host "[INFO] OHIF port $OhifPort is already in use. Reusing the existing service." -ForegroundColor Yellow
    Write-Host '       (its logs stay in the terminal that started it)' -ForegroundColor DarkGray
} else {
    Write-Host "[2/2] Starting OHIF viewer on port $OhifPort..." -ForegroundColor Green
    New-EmptyLog $OhifLog
    # Pass OHIF_PORT so the webpack dev server listens on $OhifPort
    # (see ohif-viewer\platform\app\.webpack\webpack.pwa.js: OHIF_PORT = process.env.OHIF_PORT || 3000)
    $ohifCmd = "`$env:OHIF_PORT='$OhifPort'; npm run dev 2>&1 | " +
        "Out-File -FilePath '$OhifLog' -Append -Encoding utf8"
    $proc = Start-Process powershell.exe -WorkingDirectory $OhifDir -WindowStyle Hidden -PassThru -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $ohifCmd
    )
    $startedPids += $proc.Id
}

if ($startedPids) {
    Set-Content -Path $PidFile -Value $startedPids -Encoding ascii
}

Write-Host ''
Write-Host 'Waiting for services...' -ForegroundColor Yellow
$ready = $false
for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep -Seconds 1
    if ((Test-Port 5000) -and (Test-Port $OhifPort)) {
        $ready = $true
        break
    }
}

if (-not $ready) {
    Write-Host 'Services are not ready yet. Check the logs below for errors.' -ForegroundColor Yellow
    Write-Host 'Flask: http://127.0.0.1:5000'
    Write-Host "OHIF:  http://127.0.0.1:$OhifPort"
} else {
    Write-Host ''
    Write-Host 'System started successfully.' -ForegroundColor Green
    Write-Host 'Homepage: http://127.0.0.1:5000' -ForegroundColor White
    Write-Host "OHIF:     http://127.0.0.1:$OhifPort" -ForegroundColor White
    Write-Host ''
    # Use 127.0.0.1 instead of localhost: on Windows 'localhost' resolves to ::1 first,
    # and Flask only listens on IPv4, causing a ~2s fallback delay per request.
    Start-Process 'http://127.0.0.1:5000'
}

# Make sure both log files exist so the tail loop does not throw.
foreach ($log in $FlaskLog, $OhifLog) {
    if (-not (Test-Path $log)) { New-EmptyLog $log }
}

Write-Host ''
Write-Host '=== Live logs (Ctrl+C stops the tail only, services keep running) ===' -ForegroundColor Cyan
Write-Host '=== Stop services with: .\start_radar_system.ps1 -Stop ===' -ForegroundColor DarkGray
Write-Host ''
Watch-Logs -Paths $FlaskLog, $OhifLog -Prefixes 'flask', 'ohif' -Colors 'Cyan', 'Magenta'
