# install.ps1 - Semantic Compressor setup script for Windows / PowerShell.
#
# Creates a .venv, installs requirements.txt, and runs a quick sanity test
# (tests/test_profiler.py) to verify the environment is functional.
#
# Usage:
#   .\install.ps1
#
# Exit codes:
#   0 - success
#   1 - Python 3.11+ not found
#   2 - venv creation failed
#   3 - pip install failed
#   4 - sanity test failed

$ErrorActionPreference = "Stop"

Write-Host "=== Semantic Compressor - Windows installer ===" -ForegroundColor Cyan

# -----------------------------------------------------------------------------
# 1. Locate a Python 3.11+ interpreter
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "[1/5] Looking for Python 3.11+..." -ForegroundColor Yellow

function Get-PythonVersion {
    param([string]$Exe)
    try {
        $output = & $Exe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $output) {
            return $output.Trim()
        }
    } catch {
        return $null
    }
    return $null
}

function Test-PythonOk {
    param([string]$Version)
    if (-not $Version) { return $false }
    $parts = $Version.Split('.')
    if ($parts.Count -lt 2) { return $false }
    $major = [int]$parts[0]
    $minor = [int]$parts[1]
    return ($major -gt 3) -or ($major -eq 3 -and $minor -ge 11)
}

$pythonExe = $null

# Probe known candidates in order of preference (3.13 first).
$candidates = @("python3.13", "python3.12", "python3.11", "python3", "python", "py")
foreach ($candidate in $candidates) {
    $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($null -eq $cmd) { continue }

    # `py` launcher needs an explicit version flag.
    if ($candidate -eq "py") {
        foreach ($flag in @("-3.13", "-3.12", "-3.11")) {
            $version = & py $flag -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($LASTEXITCODE -eq 0 -and (Test-PythonOk $version)) {
                $pythonExe = "py $flag"
                Write-Host "  Found Python $version via 'py $flag'" -ForegroundColor Green
                break
            }
        }
        if ($pythonExe) { break }
        continue
    }

    $version = Get-PythonVersion -Exe $candidate
    if (Test-PythonOk $version) {
        $pythonExe = $candidate
        Write-Host "  Found Python $version at '$candidate'" -ForegroundColor Green
        break
    }
}

if (-not $pythonExe) {
    Write-Host "ERROR: No Python 3.11+ interpreter found on PATH." -ForegroundColor Red
    Write-Host "Install Python from https://www.python.org/downloads/ and re-run." -ForegroundColor Red
    exit 1
}

# -----------------------------------------------------------------------------
# 2. Create venv if absent
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "[2/5] Preparing .venv..." -ForegroundColor Yellow

$venvPath = Join-Path $PSScriptRoot ".venv"
if (Test-Path $venvPath) {
    Write-Host "  .venv already exists, reusing it." -ForegroundColor Green
} else {
    Write-Host "  Creating .venv with $pythonExe..."
    # Splat the 'py -3.X' case where pythonExe contains a space.
    $parts = $pythonExe.Split(' ')
    if ($parts.Count -gt 1) {
        & $parts[0] $parts[1] -m venv $venvPath
    } else {
        & $pythonExe -m venv $venvPath
    }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPath)) {
        Write-Host "ERROR: Failed to create virtual environment at $venvPath." -ForegroundColor Red
        exit 2
    }
    Write-Host "  Created $venvPath" -ForegroundColor Green
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "ERROR: venv python not found at $venvPython." -ForegroundColor Red
    exit 2
}

# -----------------------------------------------------------------------------
# 3. Upgrade pip
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "[3/5] Upgrading pip..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip upgrade failed." -ForegroundColor Red
    exit 3
}
Write-Host "  pip is up to date." -ForegroundColor Green

# -----------------------------------------------------------------------------
# 4. Install requirements
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "[4/5] Installing requirements.txt..." -ForegroundColor Yellow
$reqPath = Join-Path $PSScriptRoot "requirements.txt"
if (-not (Test-Path $reqPath)) {
    Write-Host "ERROR: requirements.txt not found at $reqPath." -ForegroundColor Red
    exit 3
}
& $venvPython -m pip install -r $reqPath
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip install -r requirements.txt failed." -ForegroundColor Red
    exit 3
}
Write-Host "  Requirements installed." -ForegroundColor Green

# -----------------------------------------------------------------------------
# 5. Quick sanity test
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "[5/5] Running sanity test (tests/test_profiler.py)..." -ForegroundColor Yellow
& $venvPython -m pytest tests/test_profiler.py -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Sanity test failed. The environment is installed but tests do not pass." -ForegroundColor Red
    exit 4
}
Write-Host "  Sanity test passed." -ForegroundColor Green

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Install complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Activate the venv:        .\.venv\Scripts\Activate.ps1"
Write-Host "  2. Run the end-to-end POC:   python examples\run_poc.py"
Write-Host "  3. Run the full test suite:  python -m pytest"
Write-Host ""
exit 0
